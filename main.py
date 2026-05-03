"""
Echoes AI — Railway API Server
FastAPI backend for public character library, moderation, and image generation.

Endpoints:
  GET  /health                    — health check
  POST /characters/upload         — upload a public character
  GET  /characters/discover       — browse approved public characters
  GET  /characters/search         — search by name or tags
  POST /characters/like           — increment like count
  POST /characters/chat           — increment chat count + creator earnings
  GET  /characters/{id}           — get single character
  POST /moderate                  — run moderation check on text
  POST /images/generate           — generate character art (Stable Diffusion)
  POST /users/sync                — sync user account from app
  GET  /creators/{id}/earnings    — get creator shard earnings
"""

import os
import uuid
import base64
import httpx
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# ── INIT ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Echoes AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
PORT         = int(os.environ.get("PORT", "8000"))

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Creator earnings rate — 1% of shards spent
CREATOR_RATE = 0.01

# ── MODELS ────────────────────────────────────────────────────────────────────

class CharacterUpload(BaseModel):
    id: str
    name: str
    creator_id: str
    setting: str
    art_url: Optional[str] = None
    worldview: Optional[str] = None
    speech_pattern: Optional[str] = None
    emotional_baseline: Optional[str] = None
    relationship_dynamic: Optional[str] = None
    origin: Optional[str] = None
    deepest_want: Optional[str] = None
    deepest_fear: Optional[str] = None
    secret: Optional[str] = None
    affection_expression: Optional[str] = None
    anger_expression: Optional[str] = None
    verbal_quirks: Optional[str] = None
    hard_limits: Optional[str] = None
    default_relationship_view: Optional[str] = None
    art_style: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[str] = None
    archetype: Optional[str] = "none"
    tags: Optional[List[str]] = []

class ModerationRequest(BaseModel):
    content: str
    character_id: Optional[str] = None

class ImageGenRequest(BaseModel):
    character_id: str
    creator_id: str
    description: str
    name: str
    art_style: str  # "anime" or "realistic"

class ChatCountRequest(BaseModel):
    character_id: str
    creator_id: str
    shards_spent: int = 0

class LikeRequest(BaseModel):
    character_id: str

class UserSync(BaseModel):
    id: str
    username: Optional[str] = None
    is_subscriber: bool = False
    free_week_end: int = 0

# ── HEALTH ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Echoes AI API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

# ── CHARACTER UPLOAD ──────────────────────────────────────────────────────────

@app.post("/characters/upload")
async def upload_character(char: CharacterUpload):
    """
    Upload a character to the public library.
    Goes into moderation queue first — not immediately visible.
    """

    # Basic server-side moderation check
    content_to_check = " ".join(filter(None, [
        char.name, char.worldview, char.origin,
        char.deepest_want, char.deepest_fear, char.secret,
        char.hard_limits
    ]))

    if not passes_basic_moderation(content_to_check):
        raise HTTPException(status_code=400, detail="Content failed moderation check.")

    # Insert into characters table (not approved yet)
    char_data = {
        "id":                       char.id,
        "name":                     char.name,
        "creator_id":               char.creator_id,
        "setting":                  char.setting,
        "art_url":                  char.art_url,
        "worldview":                char.worldview,
        "speech_pattern":           char.speech_pattern,
        "emotional_baseline":       char.emotional_baseline,
        "relationship_dynamic":     char.relationship_dynamic,
        "origin":                   char.origin,
        "deepest_want":             char.deepest_want,
        "deepest_fear":             char.deepest_fear,
        "secret":                   char.secret,
        "affection_expression":     char.affection_expression,
        "anger_expression":         char.anger_expression,
        "verbal_quirks":            char.verbal_quirks,
        "hard_limits":              char.hard_limits,
        "default_relationship_view": char.default_relationship_view,
        "art_style":                char.art_style,
        "gender":                   char.gender,
        "age":                      char.age,
        "archetype":                char.archetype,
        "tags":                     char.tags,
        "is_public":                True,
        "approved":                 False,  # starts in queue
        "likes":                    0,
        "total_chats":              0,
    }

    result = supabase.table("characters").insert(char_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to save character.")

    # Add to moderation queue
    supabase.table("moderation_queue").insert({
        "character_id": char.id,
        "status": "pending"
    }).execute()

    return {"success": True, "character_id": char.id, "status": "pending_moderation"}

# ── DISCOVER ──────────────────────────────────────────────────────────────────

@app.get("/characters/discover")
async def discover_characters(
    limit: int = 20,
    offset: int = 0,
    sort: str = "newest"  # newest | popular | trending
):
    """Browse approved public characters."""

    order_col = "created_at" if sort == "newest" else "likes"

    result = supabase.table("characters") \
        .select("id, name, creator_id, setting, art_url, art_style, gender, age, worldview, archetype, tags, likes, total_chats, created_at") \
        .eq("is_public", True) \
        .eq("approved", True) \
        .eq("deleted", False) \
        .order(order_col, desc=True) \
        .range(offset, offset + limit - 1) \
        .execute()

    return {"characters": result.data, "count": len(result.data)}

# ── SEARCH ────────────────────────────────────────────────────────────────────

@app.get("/characters/search")
async def search_characters(
    q: str = "",
    setting: str = "",
    art_style: str = "",
    limit: int = 20
):
    """Search public characters by name, setting, or art style."""

    query = supabase.table("characters") \
        .select("id, name, creator_id, setting, art_url, art_style, gender, age, worldview, archetype, tags, likes, total_chats") \
        .eq("is_public", True) \
        .eq("approved", True) \
        .eq("deleted", False)

    if q:
        query = query.ilike("name", f"%{q}%")
    if setting:
        query = query.eq("setting", setting)
    if art_style:
        query = query.eq("art_style", art_style)

    result = query.order("likes", desc=True).limit(limit).execute()

    return {"characters": result.data, "count": len(result.data)}

# ── SINGLE CHARACTER ──────────────────────────────────────────────────────────

@app.get("/characters/{character_id}")
async def get_character(character_id: str):
    """Get a single approved public character by ID."""

    result = supabase.table("characters") \
        .select("*") \
        .eq("id", character_id) \
        .eq("is_public", True) \
        .eq("approved", True) \
        .single() \
        .execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Character not found.")

    return result.data

# ── LIKE ──────────────────────────────────────────────────────────────────────

@app.post("/characters/like")
async def like_character(req: LikeRequest):
    """Increment like count on a character."""

    supabase.rpc("increment_likes", {"char_id": req.character_id}).execute()
    return {"success": True}

# ── CHAT COUNT + CREATOR EARNINGS ─────────────────────────────────────────────

@app.post("/characters/chat")
async def record_chat(req: ChatCountRequest):
    """
    Record a chat interaction on a public character.
    Increments total_chats and adds creator earnings.
    """

    # Increment chat count
    supabase.rpc("increment_chats", {"char_id": req.character_id}).execute()

    # Calculate creator earnings
    # 1% of shards spent, or 1% of "1 shard equivalent" per chat
    shard_equivalent = req.shards_spent if req.shards_spent > 0 else 1
    earnings_fraction = shard_equivalent * CREATOR_RATE

    # Upsert creator earnings
    existing = supabase.table("creator_earnings") \
        .select("pending_fraction, total_shards_earned") \
        .eq("creator_id", req.creator_id) \
        .execute()

    if existing.data:
        current_fraction = existing.data[0]["pending_fraction"] + earnings_fraction
        whole_shards = int(current_fraction)
        new_fraction = current_fraction - whole_shards
        new_total = existing.data[0]["total_shards_earned"] + whole_shards

        supabase.table("creator_earnings").update({
            "pending_fraction":    new_fraction,
            "total_shards_earned": new_total,
            "updated_at":          datetime.utcnow().isoformat()
        }).eq("creator_id", req.creator_id).execute()

    else:
        whole_shards = int(earnings_fraction)
        supabase.table("creator_earnings").insert({
            "creator_id":          req.creator_id,
            "pending_fraction":    earnings_fraction - whole_shards,
            "total_shards_earned": whole_shards,
        }).execute()

    return {"success": True, "shards_earned": whole_shards if existing.data else int(earnings_fraction)}

# ── CREATOR EARNINGS ──────────────────────────────────────────────────────────

@app.get("/creators/{creator_id}/earnings")
async def get_creator_earnings(creator_id: str):
    """Get creator shard earnings."""

    result = supabase.table("creator_earnings") \
        .select("*") \
        .eq("creator_id", creator_id) \
        .execute()

    if not result.data:
        return {"creator_id": creator_id, "total_shards_earned": 0, "pending_fraction": 0}

    return result.data[0]

# ── IMAGE GENERATION ──────────────────────────────────────────────────────────

@app.post("/images/generate")
async def generate_image(req: ImageGenRequest):
    """
    Generate character art using Stable Diffusion.
    Returns the image URL after uploading to Supabase Storage.

    Note: Requires a Stable Diffusion instance running.
    For now returns a placeholder — full implementation
    requires SD WebUI API running as a separate Railway service.
    """

    # Build the image prompt from character description
    style_prefix = "anime art style, digital illustration," \
        if req.art_style.lower() == "anime" \
        else "photorealistic, detailed portrait photography,"

    prompt = (
        f"{style_prefix} character portrait of {req.name}, "
        f"{req.description}, "
        f"dark moody background, dramatic lighting, high quality, detailed"
    )

    negative_prompt = (
        "blurry, low quality, watermark, text, deformed, "
        "extra limbs, bad anatomy, ugly"
    )

    # Try to call Stable Diffusion WebUI API if available
    sd_url = os.environ.get("SD_API_URL", "")

    if sd_url:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{sd_url}/sdapi/v1/txt2img",
                    json={
                        "prompt":          prompt,
                        "negative_prompt": negative_prompt,
                        "steps":           25,
                        "width":           512,
                        "height":          768,
                        "cfg_scale":       7,
                        "sampler_name":    "DPM++ 2M Karras"
                    }
                )
                data = response.json()
                image_b64 = data["images"][0]
                image_bytes = base64.b64decode(image_b64)

                # Upload to Supabase Storage
                file_path = f"{req.creator_id}/{req.character_id}.png"
                supabase.storage.from_("character-art").upload(
                    file_path,
                    image_bytes,
                    {"content-type": "image/png"}
                )

                public_url = supabase.storage.from_("character-art").get_public_url(file_path)

                return {"success": True, "art_url": public_url}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

    else:
        # SD not configured yet — return placeholder
        return {
            "success": False,
            "art_url": None,
            "message": "Image generation service not configured yet. Upload an image instead."
        }

# ── USER SYNC ─────────────────────────────────────────────────────────────────

@app.post("/users/sync")
async def sync_user(user: UserSync):
    """Sync user account from app to server."""

    user_data = {
        "id":             user.id,
        "username":       user.username,
        "is_subscriber":  user.is_subscriber,
        "free_week_end":  user.free_week_end,
    }

    supabase.table("users").upsert(user_data).execute()
    return {"success": True}

# ── MODERATION ────────────────────────────────────────────────────────────────

@app.post("/moderate")
async def moderate_content(req: ModerationRequest):
    """
    Server-side moderation check.
    Second pass after on-device Gemma moderation.
    Uses keyword filtering for now — can swap in an ML model later.
    """

    result = passes_basic_moderation(req.content)

    # If this is for a queued character, update its status
    if req.character_id:
        if result:
            supabase.table("moderation_queue").update({
                "status":      "approved",
                "reviewed_at": datetime.utcnow().isoformat()
            }).eq("character_id", req.character_id).execute()

            supabase.table("characters").update({
                "approved": True
            }).eq("id", req.character_id).execute()
        else:
            supabase.table("moderation_queue").update({
                "status":           "rejected",
                "reviewed_at":      datetime.utcnow().isoformat(),
                "rejection_reason": "Content policy violation"
            }).eq("character_id", req.character_id).execute()

    return {"passes": result}

# ── MODERATION HELPER ─────────────────────────────────────────────────────────

BLOCKED_TERMS = [
    "child", "minor", "underage", "loli", "shota",
    "kill", "murder", "suicide", "rape", "torture",
    "nazi", "terrorist", "bomb"
]

def passes_basic_moderation(content: str) -> bool:
    """Basic keyword moderation — catches obvious violations."""
    content_lower = content.lower()
    for term in BLOCKED_TERMS:
        if term in content_lower:
            return False
    return True

# ── DATABASE HELPERS ──────────────────────────────────────────────────────────
# Run these SQL functions once in Supabase SQL Editor:
#
# create or replace function increment_likes(char_id text)
# returns void as $$
#   update characters set likes = likes + 1 where id = char_id;
# $$ language sql;
#
# create or replace function increment_chats(char_id text)
# returns void as $$
#   update characters set total_chats = total_chats + 1 where id = char_id;
# $$ language sql;

# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
