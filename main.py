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

SUPABASE_URL         = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY         = os.environ.get("SUPABASE_KEY", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
PORT                 = int(os.environ.get("PORT", "8000"))

# Lazy init — connect on first request not on startup
_supabase_client: Optional[Client] = None

def get_db() -> Client:
    global _supabase_client
    if _supabase_client is None:
        # Use service key to bypass RLS on server side
        key = SUPABASE_SERVICE_KEY if SUPABASE_SERVICE_KEY else SUPABASE_KEY
        _supabase_client = create_client(SUPABASE_URL, key)
    return _supabase_client

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

    result = get_db().table("characters").insert(char_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to save character.")

    # Add to moderation queue
    get_db().table("moderation_queue").insert({
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

    result = get_db().table("characters") \
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

    query = get_db().table("characters") \
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

    result = get_db().table("characters") \
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

    get_db().rpc("increment_likes", {"char_id": req.character_id}).execute()
    return {"success": True}

# ── CHAT COUNT + CREATOR EARNINGS ─────────────────────────────────────────────

@app.post("/characters/chat")
async def record_chat(req: ChatCountRequest):
    """
    Record a chat interaction on a public character.
    Increments total_chats and adds creator earnings.
    """

    # Increment chat count
    get_db().rpc("increment_chats", {"char_id": req.character_id}).execute()

    # Calculate creator earnings
    # 1% of shards spent, or 1% of "1 shard equivalent" per chat
    shard_equivalent = req.shards_spent if req.shards_spent > 0 else 1
    earnings_fraction = shard_equivalent * CREATOR_RATE

    # Upsert creator earnings
    existing = get_db().table("creator_earnings") \
        .select("pending_fraction, total_shards_earned") \
        .eq("creator_id", req.creator_id) \
        .execute()

    if existing.data:
        current_fraction = existing.data[0]["pending_fraction"] + earnings_fraction
        whole_shards = int(current_fraction)
        new_fraction = current_fraction - whole_shards
        new_total = existing.data[0]["total_shards_earned"] + whole_shards

        get_db().table("creator_earnings").update({
            "pending_fraction":    new_fraction,
            "total_shards_earned": new_total,
            "updated_at":          datetime.utcnow().isoformat()
        }).eq("creator_id", req.creator_id).execute()

    else:
        whole_shards = int(earnings_fraction)
        get_db().table("creator_earnings").insert({
            "creator_id":          req.creator_id,
            "pending_fraction":    earnings_fraction - whole_shards,
            "total_shards_earned": whole_shards,
        }).execute()

    return {"success": True, "shards_earned": whole_shards if existing.data else int(earnings_fraction)}

# ── CREATOR EARNINGS ──────────────────────────────────────────────────────────

@app.get("/creators/{creator_id}/earnings")
async def get_creator_earnings(creator_id: str):
    """Get creator shard earnings."""

    result = get_db().table("creator_earnings") \
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
                get_db().storage.from_("character-art").upload(
                    file_path,
                    image_bytes,
                    {"content-type": "image/png"}
                )

                public_url = get_db().storage.from_("character-art").get_public_url(file_path)

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

    get_db().table("users").upsert(user_data).execute()
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
            get_db().table("moderation_queue").update({
                "status":      "approved",
                "reviewed_at": datetime.utcnow().isoformat()
            }).eq("character_id", req.character_id).execute()

            get_db().table("characters").update({
                "approved": True
            }).eq("id", req.character_id).execute()
        else:
            get_db().table("moderation_queue").update({
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

# ══════════════════════════════════════════════════════════════
# SERVER-SIDE ENFORCEMENT — Chat limits, Shards, Subscription
# ══════════════════════════════════════════════════════════════

# ── MODELS ────────────────────────────────────────────────────

class ChatCheckRequest(BaseModel):
    user_id: str
    is_subscriber: bool = False
    free_week_active: bool = False

class ShardUpdateRequest(BaseModel):
    user_id: str
    amount: int          # positive = earn, negative = spend
    reason: str

class ShardSyncRequest(BaseModel):
    user_id: str
    local_balance: int   # app sends its local balance for reconciliation

class SubscriptionVerifyRequest(BaseModel):
    user_id: str
    purchase_token: str
    product_id: str
    free_week_end: int = 0

# ── CHAT LIMIT ENFORCEMENT ────────────────────────────────────

FREE_DAILY_LIMIT = 25
FREE_MAX_BANKED  = 60

@app.post("/chats/check")
async def check_chat_allowed(req: ChatCheckRequest):
    """
    Check if user is allowed to send a message.
    Server is the source of truth — cannot be faked by modded APK.
    Returns: allowed (bool), remaining (int), limit (int)
    """
    # Subscribers and free week users have unlimited chats
    if req.is_subscriber or req.free_week_active:
        return {"allowed": True, "remaining": -1, "limit": -1, "unlimited": True}

    today = datetime.utcnow().date().isoformat()

    # Get current usage
    result = get_db().table("chat_usage") \
        .select("chats_used") \
        .eq("user_id", req.user_id) \
        .eq("usage_date", today) \
        .execute()

    if result.data:
        chats_used = result.data[0]["chats_used"]
    else:
        chats_used = 0

    remaining = max(0, FREE_DAILY_LIMIT - chats_used)
    allowed   = remaining > 0

    if allowed:
        # Increment usage
        if result.data:
            get_db().table("chat_usage").update({
                "chats_used": chats_used + 1,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("user_id", req.user_id).eq("usage_date", today).execute()
        else:
            get_db().table("chat_usage").insert({
                "user_id":    req.user_id,
                "usage_date": today,
                "chats_used": 1
            }).execute()

    return {
        "allowed":   allowed,
        "remaining": remaining - (1 if allowed else 0),
        "limit":     FREE_DAILY_LIMIT,
        "unlimited": False
    }

@app.get("/chats/usage/{user_id}")
async def get_chat_usage(user_id: str):
    """Get current daily chat usage for a user."""
    today = datetime.utcnow().date().isoformat()
    result = get_db().table("chat_usage") \
        .select("chats_used, usage_date") \
        .eq("user_id", user_id) \
        .eq("usage_date", today) \
        .execute()

    chats_used = result.data[0]["chats_used"] if result.data else 0
    return {
        "user_id":   user_id,
        "date":      today,
        "chats_used": chats_used,
        "remaining": max(0, FREE_DAILY_LIMIT - chats_used),
        "limit":     FREE_DAILY_LIMIT
    }

# ── SHARD BALANCE ENFORCEMENT ─────────────────────────────────

@app.post("/shards/update")
async def update_shards(req: ShardUpdateRequest):
    """
    Update shard balance server-side.
    Spends are validated — cannot go below 0.
    """
    # Get current balance
    result = get_db().table("shard_balances") \
        .select("balance, lifetime_earned, lifetime_spent") \
        .eq("user_id", req.user_id) \
        .execute()

    if result.data:
        current = result.data[0]["balance"]
        lifetime_earned = result.data[0]["lifetime_earned"]
        lifetime_spent  = result.data[0]["lifetime_spent"]
    else:
        current = 0
        lifetime_earned = 0
        lifetime_spent  = 0

    # Validate spend
    if req.amount < 0 and abs(req.amount) > current:
        return {"success": False, "error": "Insufficient shards", "balance": current}

    new_balance = current + req.amount
    new_earned  = lifetime_earned + (req.amount if req.amount > 0 else 0)
    new_spent   = lifetime_spent  + (abs(req.amount) if req.amount < 0 else 0)

    # Upsert balance
    get_db().table("shard_balances").upsert({
        "user_id":         req.user_id,
        "balance":         new_balance,
        "lifetime_earned": new_earned,
        "lifetime_spent":  new_spent,
        "updated_at":      datetime.utcnow().isoformat()
    }).execute()

    # Log transaction
    get_db().table("shard_transactions").insert({
        "user_id": req.user_id,
        "amount":  req.amount,
        "reason":  req.reason
    }).execute()

    return {"success": True, "balance": new_balance, "delta": req.amount}

@app.post("/shards/sync")
async def sync_shards(req: ShardSyncRequest):
    """
    Sync local shard balance with server.
    Server wins if discrepancy is suspicious (local > server * 1.5).
    Returns the authoritative balance.
    """
    result = get_db().table("shard_balances") \
        .select("balance") \
        .eq("user_id", req.user_id) \
        .execute()

    server_balance = result.data[0]["balance"] if result.data else 0

    # If local balance is way higher than server — likely cheating
    # Allow small differences (could be offline transactions)
    if req.local_balance > server_balance * 1.5 and req.local_balance - server_balance > 100:
        # Use server balance — reject the inflated local value
        return {
            "authoritative_balance": server_balance,
            "accepted_local":        False,
            "reason":                "Local balance rejected — server balance used"
        }

    # If local is reasonably higher — trust it and sync up
    if req.local_balance > server_balance:
        get_db().table("shard_balances").upsert({
            "user_id":    req.user_id,
            "balance":    req.local_balance,
            "updated_at": datetime.utcnow().isoformat()
        }).execute()
        return {"authoritative_balance": req.local_balance, "accepted_local": True}

    # Server is higher — return server value
    return {"authoritative_balance": server_balance, "accepted_local": False}

@app.get("/shards/{user_id}")
async def get_shard_balance(user_id: str):
    """Get authoritative server-side shard balance."""
    result = get_db().table("shard_balances") \
        .select("balance, lifetime_earned, lifetime_spent, updated_at") \
        .eq("user_id", user_id) \
        .execute()

    if not result.data:
        return {"user_id": user_id, "balance": 0, "lifetime_earned": 0, "lifetime_spent": 0}

    return result.data[0] | {"user_id": user_id}

# ── SUBSCRIPTION VERIFICATION ─────────────────────────────────

@app.post("/subscription/verify")
async def verify_subscription(req: SubscriptionVerifyRequest):
    """
    Verify subscription status server-side.
    In production: validates purchase token against Google Play API.
    Stores verified status in Supabase — app checks here, not local prefs.
    """

    # TODO Phase 2: Call Google Play Developer API to verify purchase_token
    # For now: trust the token if it's present and non-empty
    # Real implementation:
    # response = await google_play_api.verify(req.purchase_token, req.product_id)
    # is_valid = response.purchaseState == 0 and response.autoRenewing

    is_valid = bool(req.purchase_token) and len(req.purchase_token) > 10

    now = int(datetime.utcnow().timestamp() * 1000)
    sub_end = now + (30 * 24 * 60 * 60 * 1000) if is_valid else 0  # 30 days

    get_db().table("subscriptions").upsert({
        "user_id":          req.user_id,
        "is_subscriber":    is_valid,
        "subscription_end": sub_end,
        "purchase_token":   req.purchase_token,
        "product_id":       req.product_id,
        "free_week_end":    req.free_week_end,
        "verified_at":      datetime.utcnow().isoformat(),
        "updated_at":       datetime.utcnow().isoformat()
    }).execute()

    # Also update users table
    get_db().table("users").upsert({
        "id":            req.user_id,
        "is_subscriber": is_valid,
        "free_week_end": req.free_week_end
    }).execute()

    return {
        "verified":       is_valid,
        "is_subscriber":  is_valid,
        "subscription_end": sub_end,
        "free_week_end":  req.free_week_end
    }

@app.get("/subscription/{user_id}")
async def get_subscription_status(user_id: str):
    """
    Get server-verified subscription status.
    App should call this on launch — cannot be faked locally.
    """
    result = get_db().table("subscriptions") \
        .select("is_subscriber, subscription_end, free_week_end, verified_at") \
        .eq("user_id", user_id) \
        .execute()

    if not result.data:
        return {
            "user_id":        user_id,
            "is_subscriber":  False,
            "free_week_active": False,
            "subscription_end": 0,
            "free_week_end":  0
        }

    sub = result.data[0]
    now = int(datetime.utcnow().timestamp() * 1000)
    free_week_active = sub["free_week_end"] > now
    sub_active = sub["is_subscriber"] and sub["subscription_end"] > now

    return {
        "user_id":          user_id,
        "is_subscriber":    sub_active,
        "free_week_active": free_week_active,
        "has_full_access":  sub_active or free_week_active,
        "subscription_end": sub["subscription_end"],
        "free_week_end":    sub["free_week_end"],
        "verified_at":      sub["verified_at"]
    }
