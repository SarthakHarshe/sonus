# Sonus

Describe a song in a sentence and get back an actual track: audio, lyrics, cover art, and genre tags.

Type "80s synth-pop about missing the last train home", hit create, and a couple of minutes later there are two finished songs sitting in your library. You can also go the other way and hand it your own lyrics and style tags if you already know what you want.

## What happens when you hit create

```
Create page ──► server action: write two Song rows (status "queued")
                              │
                              └─► Inngest "generate-song-event", one job at a time per user
                                          │
                                          ├─ credits check
                                          ├─ pick a Modal endpoint from what you filled in
                                          └─► POST to Modal (L40S GPU)
                                                    │
                                                    ├─ Qwen2-7B: description ─► audio tags
                                                    ├─ Qwen2-7B: description ─► structured lyrics
                                                    ├─ ACE-Step: tags + lyrics ─► 3 minute WAV
                                                    ├─ SDXL-Turbo: album cover
                                                    ├─ Qwen2-7B: 3 to 5 genre categories
                                                    └─ upload WAV + PNG to S3, return the keys
                                          │
                                          └─ save keys, connect categories, deduct 1 credit
```

Every request is queued twice, once at guidance scale 7.5 and once at 15. The model is stochastic and guidance changes how literally it follows the prompt, so you get two takes on the same idea and keep the one you like. Each one is its own job and its own credit, so a single create costs 2.

### Three ways in

The create panel has a simple mode and a custom mode, and the backend has a matching endpoint for each shape of input:

| You give it | Endpoint | What the LLM fills in |
|---|---|---|
| A full description of the song | `generate_from_description` | Both the audio tags and the lyrics |
| Style tags plus your own lyrics | `generate_with_lyrics` | Nothing, it just generates |
| Style tags plus a description of the lyrics | `generate_with_described_lyrics` | The lyrics |

There is an instrumental switch too. Flip it and the lyrics step is skipped entirely, with `[instrumental]` passed to the model instead.

The prompt templates are in `backend/prompts.py`. One reshapes loose English into the comma-separated tag format ACE-Step expects (genre, vocal type, instruments, mood, tempo, key). The other writes lyrics with `[verse]`, `[chorus]`, and `[bridge]` structure tags, because ACE-Step uses those to shape the arrangement.

## The rest of the app

**Home** is a feed of songs people have published: a trending row (published in the last two days), then shelves grouped by primary category. You can like tracks and each play bumps a listen count.

**Create** is a two panel view. The song panel on the left, your track list on the right. Tracks can be renamed, published, downloaded, or played. Generation status shows up per track, and there is a refresh to re-check it (there is no live polling yet, so a queued track will not update on its own).

**The player** is a global bottom bar backed by a small zustand store, so audio keeps playing while you move around the app.

## Repo layout

```
frontend/    Next.js 15 app (T3 stack), Better Auth, Prisma, Inngest, Polar
backend/     Modal app in main.py, prompt templates in prompts.py
  └── ACE-Step/   git submodule pointing at the upstream ACE-Step repo
```

The submodule is there for reference. The Modal image does not use it, it clones and pip installs ACE-Step directly during the image build.

## Stack

| Layer | What it is |
|---|---|
| Frontend | Next.js 15 (App Router), React 19, TypeScript, Tailwind v4, shadcn/ui |
| Auth | Better Auth with email and password, plus the Better Auth UI components |
| Data | PostgreSQL via Prisma |
| Queue | Inngest, limited to one generation per user at a time |
| Compute | Modal, NVIDIA L40S, endpoints behind Modal proxy auth |
| Music | [ACE-Step](https://github.com/ace-step/ACE-Step) |
| Text | Qwen2-7B-Instruct for tags, lyrics, and categories |
| Cover art | SDXL-Turbo, 2 inference steps |
| Storage | S3, served to the browser through presigned URLs |
| Payments | Polar (currently pointed at their sandbox) |

Model weights live on two Modal volumes (`ace-step-models` and `qwen-hf-cache`) so containers do not re-download several gigabytes on every cold start.

## Running it

You will need Node 20+, Python 3.12, PostgreSQL, an S3 bucket, a Modal account, and a Polar sandbox account.

### Backend

```bash
cd backend
pip install -r requirements.txt
modal deploy main.py
```

Create a Modal secret named `sonus-secret` first, holding `S3_BUCKET_NAME` and AWS credentials that can write to that bucket.

The endpoints use Modal's proxy auth (`requires_proxy_auth=True`), so calls need `Modal-Key` and `Modal-Secret` headers. Deploy prints one URL per endpoint. You will need three of them for the frontend env.

### Frontend

```bash
cd frontend
npm install
./start-database.sh        # optional, spins up Postgres in Docker
npx prisma migrate deploy
npm run dev
```

Environment variables, all validated in `src/env.js`:

```
DATABASE_URL
BETTER_AUTH_SECRET
MODAL_KEY
MODAL_SECRET
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY_ID        # note the trailing _ID, that is the actual name
AWS_REGION
S3_BUCKET_NAME
GENERATE_FROM_DESCRIPTION       # the three deployed Modal endpoint URLs
GENERATE_FROM_DESCRIBED_LYRICS
GENERATE_WITH_LYRICS
POLAR_ACCESS_TOKEN
POLAR_WEBHOOK_SECRET
```

Run the Inngest dev server alongside `npm run dev` so jobs actually run. For testing Polar webhooks locally there is `npm run polar-webhooks`, which just opens an ngrok tunnel to port 3000.

## Credits and billing

New accounts start with 100 credits and each generated song costs 1, taken only after the generation succeeds. Buying more goes through Polar checkout, and the credit top-up happens in the `onOrderPaid` webhook.

Worth knowing: the Polar client is hardcoded to `server: "sandbox"` and the three product IDs are hardcoded in `src/lib/auth.ts` and `src/components/sidebar/upgrade.tsx`. Swap both before this touches real money.

## Known limits

- A three minute duration is fixed at queue time. The backend accepts an `audio_duration`, but the UI never exposes it.
- Track status does not update on its own, you refresh to check.
- Generation takes a few minutes per song on a cold container, and every request makes two.
- The categories come out of an LLM, so near-duplicate genre names can pile up in the category table over time.

## Credits

ACE-Step is the work of the [ACE-Step team](https://github.com/ace-step/ACE-Step) and does the actual music generation here. This project is the application layer around it.

Built by [Sarthak Harshe](https://github.com/SarthakHarshe).
