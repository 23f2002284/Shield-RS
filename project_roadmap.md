# 🗺️ Agent-as-Shield RS — Project Roadmap

> Concrete steps to go from idea → working system → publishable paper.

---

## Phase 0: Before You Write Any Code (Days 1–3)

These steps set you up for success. Skipping them leads to rework later.

### Step 0.1 — Lock Your Scope
Decide NOW what your **demo platform** is. The project doc says YouTube — stick with it.

- [ ] **Decision:** YouTube as primary platform (use YouTube Data API v3)
- [ ] **Decision:** Query type — search-based ("learn about X") not feed-based (homepage)
- [ ] **Decision:** Content type — videos only (not Shorts, not live streams)

### Step 0.2 — Set Up the Repository
```
agent-shield-rs/
├── data/                  # Raw and processed datasets
│   ├── raw/
│   ├── processed/
│   └── scrapes/
├── models/                # Saved model weights
├── src/
│   ├── scraper/           # YouTube API data collection
│   ├── manipulation/      # Module 1: Manipulation Detector
│   ├── quality/           # Module 2: Content Quality Evaluator
│   ├── optimizer/         # Module 3: Goal-Constrained Optimizer
│   ├── explainer/         # Module 4: Explainability Layer
│   ├── pipeline/          # End-to-end orchestration
│   └── api/               # FastAPI backend
├── frontend/              # Dashboard UI
├── notebooks/             # EDA and experiments
├── experiments/           # Evaluation scripts + results
├── tests/
├── requirements.txt
└── README.md
```

- [ ] Create the repo structure
- [ ] Set up a Python virtual environment (3.10+)
- [ ] Install base dependencies: `torch`, `transformers`, `sentence-transformers`, `fastapi`, `google-api-python-client`
- [ ] Get a YouTube Data API key (Google Cloud Console → enable YouTube Data API v3)

### Step 0.3 — Read the iAgent Paper
This is your closest related work. You need to understand it to differentiate.

- [ ] Read the [iAgent paper](https://aclanthology.org/) (ACL 2025 Findings)
- [ ] Read their [GitHub repo](https://github.com/) — understand their InstructRec dataset format
- [ ] Write a 1-paragraph differentiation statement for your paper intro

---

## Phase 1: Data & Baseline (Days 4–10) — *Week 1*

### Step 1.1 — Build the YouTube Scraper
You need real video metadata to work with.

```python
# What to collect per video:
{
    "video_id": str,
    "title": str,
    "description": str,
    "channel_name": str,
    "channel_id": str,
    "subscriber_count": int,      # Social proof signal (will be IGNORED in scoring)
    "view_count": int,            # Social proof signal (will be IGNORED in scoring)
    "like_count": int,            # Social proof signal (will be IGNORED in scoring)
    "duration_seconds": int,      # For time budget constraints
    "published_at": str,
    "tags": list[str],
    "category_id": int,
    "thumbnail_url": str,
    "captions_available": bool,
    "transcript": str | None      # Via youtube-transcript-api
}
```

- [ ] Write `src/scraper/youtube_scraper.py` — collect metadata for a set of queries
- [ ] Write `src/scraper/transcript_fetcher.py` — pull transcripts (use `youtube-transcript-api`)
- [ ] **Scrape 5 diverse topics** (at least 200 videos each):
  - "learn about climate change"
  - "how to invest for beginners"
  - "history of ancient Rome"
  - "machine learning tutorial"
  - "healthy meal prep"
- [ ] Store in `data/scrapes/` as JSON/CSV

> [!TIP]
> YouTube Data API has a quota of 10,000 units/day. A search costs 100 units. Plan your scraping carefully — ~100 searches/day max. Transcripts are free (no API quota).

### Step 1.2 — Download & Prepare External Datasets
- [ ] Download [Clickbait Challenge](https://www.clickbait-challenge.org/) dataset (38K headlines)
- [ ] Download [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) (58K comments, 27 emotions)
- [ ] Download [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) (credibility labels)
- [ ] Write preprocessing scripts in `notebooks/01_data_prep.ipynb`
- [ ] Store processed versions in `data/processed/`

### Step 1.3 — EDA Notebook
Build intuition about your data before modeling.

- [ ] `notebooks/02_eda.ipynb`:
  - Distribution of title lengths, ALL-CAPS usage, punctuation (!!! ???)
  - View count vs. transcript quality (is popularity ≠ quality?)
  - Emotion word frequency in titles across topics
  - Duration distributions per topic
  - How many videos have transcripts available?
- [ ] **Key question to answer:** Can you visually see the manipulation signals in the data? (You should be able to — this validates the project premise)

### Step 1.4 — Build the "Dumb Baseline"
A YouTube-native ranking to compare against later.

- [ ] `src/pipeline/baseline.py`:
  - For a query, fetch top 50 results from YouTube API (already ranked by YouTube's algorithm)
  - Record the ranking order, engagement metrics, and titles
  - This becomes your **Baseline (Human-Centric RS)** in experiments

---

## Phase 2: Core Modules (Days 11–24) — *Weeks 2–3*

Build each module independently, with its own tests.

### Step 2.1 — Module 1: Manipulation Detector (Days 11–16)

#### 2.1a — Clickbait Classifier
- [ ] `src/manipulation/clickbait.py`:
  - Fine-tune `distilbert-base-uncased` on the Clickbait Challenge dataset
  - Binary classification: clickbait (1) vs. not (0)
  - Output: `clickbait_score` ∈ [0, 1]
- [ ] Train/val/test split: 70/15/15
- [ ] **Target:** ≥ 90% F1 on test set
- [ ] Save model to `models/clickbait_distilbert/`

> [!TIP]
> **Bonus from research:** Go beyond binary — implement **tactic attribution**. Add a multi-label head that outputs a tactic vector:
> ```python
> {
>     "curiosity_gap": 0.85,    # "You won't BELIEVE..."
>     "false_urgency": 0.91,    # "BEFORE it's too late!"
>     "emotional_bait": 0.72,   # Shock/outrage framing
>     "exaggeration": 0.68      # "BEST EVER", "INSANE"
> }
> ```
> This requires manual labeling ~500 examples with tactic tags. Worth it for the explainability layer.

#### 2.1b — Emotional Manipulation Scorer
- [ ] `src/manipulation/emotion.py`:
  - Use the pre-trained GoEmotions model (`monologg/bert-base-cased-goemotions-original`)
  - Apply to video titles + first 200 words of description
  - Flag: high scores on `anger`, `fear`, `disgust`, `surprise` → manipulation signal
  - Compute `emotional_manipulation_score` = weighted sum of extreme emotion probabilities
- [ ] Test on your scraped YouTube titles — spot-check results

#### 2.1c — Social Proof Stripper
- [ ] `src/manipulation/social_proof.py`:
  - This is the simplest module — it's algorithmic, not ML
  - Function that takes video metadata and returns a **stripped version** with `view_count`, `like_count`, `subscriber_count` set to `None`
  - The downstream scorer never sees these fields
- [ ] Write unit tests proving the stripper works

#### 2.1d — Integration Test
- [ ] Feed 100 scraped videos through the full Manipulation Detector pipeline
- [ ] Manually verify: do the top-10 "most manipulative" videos *look* manipulative?
- [ ] Manually verify: do the top-10 "least manipulative" videos *look* genuine?
- [ ] Save results as `experiments/manipulation_detector_validation.csv`

---

### Step 2.2 — Module 2: Content Quality Evaluator (Days 17–22)

#### 2.2a — Information Density Scorer
- [ ] `src/quality/info_density.py`:
  - Input: video transcript + duration
  - **Step 1:** Extract named entities (spaCy `en_core_web_sm`)
  - **Step 2:** Extract factual claims (sentences with entities + verbs in declarative form)
  - **Step 3:** `info_density = unique_claims / duration_minutes`
  - Output: normalized score ∈ [0, 1]
- [ ] Test: compare a 10-min documentary vs. a 10-min vlog — density should differ significantly

#### 2.2b — Source Credibility Scorer
- [ ] `src/quality/credibility.py`:
  - Heuristic-based (no ML needed for v1):
    - `has_citations` → check description for URLs, paper references, "source:" mentions
    - `academic_language` → ratio of formal/technical terms (use a word list)
    - `balanced_framing` → presence of hedging language ("however", "on the other hand")
    - `channel_consistency` → how many videos on this topic has the channel posted?
  - Weighted sum → `credibility_score` ∈ [0, 1]
- [ ] **Explicitly NOT using:** subscriber count, view count, verified badge

#### 2.2c — Topic Coverage / Goal Alignment Scorer
- [ ] `src/quality/goal_alignment.py`:
  - Load `sentence-transformers/all-MiniLM-L6-v2`
  - Embed the user's goal query
  - Embed each video's title + description (or transcript if available)
  - `goal_alignment = cosine_similarity(query_embedding, content_embedding)`
- [ ] Test: "learn about black holes" should rank an astrophysics lecture > a sci-fi movie review

#### 2.2d — Bias/Balance Scorer
- [ ] `src/quality/bias.py`:
  - **Opinion vs. Fact ratio:** Use a simple classifier (or heuristic: subjective adjectives, first-person pronouns = opinion)
  - **Loaded language detection:** Check against NRC Emotion Lexicon
  - `bias_score` = 1.0 (perfectly balanced) → 0.0 (completely one-sided)
- [ ] Test on political/controversial topics — should flag one-sided content

#### 2.2e — Integration Test
- [ ] Feed 100 scraped videos through the full Content Quality pipeline
- [ ] Produce a ranked list by quality score
- [ ] Compare against YouTube's ranking — where do they disagree? (This is your key result)

---

### Step 2.3 — Module 3: Goal-Constrained Optimizer (Days 23–24)

#### 2.3a — Core Optimizer
- [ ] `src/optimizer/selector.py`:
  - Implement **greedy submodular maximization**:
    ```python
    def select_optimal_set(candidates, user_goal, time_budget, max_items, lambda_param=0.7):
        selected = []
        remaining_budget = time_budget
        while len(selected) < max_items and remaining_budget > 0:
            best = None
            best_score = -inf
            for c in candidates:
                if c.duration > remaining_budget:
                    continue
                quality = compute_agent_score(c)
                diversity = marginal_coverage_gain(c, selected)
                score = lambda_param * quality + (1 - lambda_param) * diversity
                if score > best_score:
                    best = c
                    best_score = score
            if best is None:
                break
            selected.append(best)
            remaining_budget -= best.duration
            candidates.remove(best)
        return selected
    ```
- [ ] `compute_agent_score()` combines all Module 1 + Module 2 scores using the inverted scoring function
- [ ] `marginal_coverage_gain()` uses embedding distance from already-selected items

#### 2.3b — Weight Tuning
- [ ] Create `src/optimizer/weights.py`:
  - Define default weights for the agent scoring function (w1–w8)
  - Allow user meta-preferences to adjust them:
    - "I want scientific sources" → boost w2 (credibility)
    - "I want diverse viewpoints" → boost w5 (bias balance)
    - "I'm short on time" → tighten time budget

---

## Phase 3: Integration & UI (Days 25–31) — *Week 5*

### Step 3.1 — End-to-End Pipeline
- [ ] `src/pipeline/agent.py`:
  ```python
  class AgentShield:
      def recommend(self, user_goal: str, time_budget: int, preferences: dict) -> RecommendationResult:
          # 1. Fetch candidates from YouTube API
          # 2. Fetch transcripts where available
          # 3. Run Manipulation Detector on all candidates
          # 4. Run Content Quality Evaluator on all candidates
          # 5. Compute agent_score for each candidate
          # 6. Run Goal-Constrained Optimizer
          # 7. Generate explanations for selected + filtered items
          # 8. Return curated set with explanations
  ```
- [ ] Write integration tests: given a query, does the full pipeline return sensible results?

### Step 3.2 — FastAPI Backend
- [ ] `src/api/main.py`:
  ```
  POST /recommend
  Body: { "goal": str, "time_budget_minutes": int, "preferences": dict }
  Response: { "recommended": [...], "filtered": [...], "metrics": {...} }
  ```
- [ ] `GET /compare` — returns both YouTube baseline and agent results side-by-side

### Step 3.3 — Dashboard UI (A/B Comparison)
Build the **split-screen A/B comparison dashboard** (from research insights):

```
┌──────────────────────┬──────────────────────┐
│   YouTube's Picks    │   Agent's Picks      │
│                      │                      │
│  1. 😱 SHOCKING...   │  1. ✅ Climate 101   │
│     47M views        │     Goal: 0.94       │
│     Clickbait: 0.91  │     Info density: Hi  │
│                      │                      │
│  2. 🔥 YOU WON'T...  │  2. ✅ Dr. Smith's   │
│     12M views        │     Credibility: 0.9  │
│     Emotional: 0.87  │     Coverage: New     │
│                      │                      │
├──────────────────────┴──────────────────────┤
│          📊 Metrics Comparison              │
│  Clickbait exposure:  72% vs 3%             │
│  Viewpoint diversity: 0.21 vs 0.84          │
│  Goal alignment:      0.45 vs 0.91          │
│  Time efficiency:     Low vs High           │
└─────────────────────────────────────────────┘
```

- [ ] Build with HTML/CSS/JS (or Streamlit for speed)
- [ ] Input: goal text, time budget slider, preference toggles
- [ ] Output: side-by-side results with per-video score breakdowns
- [ ] Include an **"Explain why filtered"** expandable section for rejected videos

### Step 3.4 — Explainability Layer
- [ ] `src/explainer/explain.py`:
  - For each **recommended** video: generate a natural-language reason
    - *"Selected: Covers fundamentals with high information density (0.87). Expert source with cited references."*
  - For each **filtered** video: generate a natural-language reason
    - *"Filtered: Clickbait title (curiosity_gap: 0.85, false_urgency: 0.91). Low information density despite 47M views."*
  - Template-based for v1; LLM-generated summaries for v2

---

## Phase 4: Evaluation & Paper (Days 32–42) — *Week 6*

### Step 4.1 — Experiment Design
Run two systems on the **same 20 queries**, compare outcomes.

- [ ] Create `experiments/queries.json` — 20 diverse queries spanning 5 topics
- [ ] For each query:
  1. Run YouTube baseline → save top-10 results
  2. Run Agent-Shield → save recommended set + filtered set
  3. Compute all metrics on both

### Step 4.2 — Compute Metrics

| Metric | How to Compute | What It Proves |
|--------|---------------|----------------|
| **Manipulation Exposure** | `avg(clickbait_score)` of recommended set | Agent filters manipulation |
| **Goal Alignment** | `avg(goal_alignment_score)` of recommended set | Agent serves the stated goal |
| **Information Density** | `avg(info_density_score)` of recommended set | Agent picks informative content |
| **Viewpoint Diversity** | Entropy of topic clusters in recommended set | Agent avoids filter bubbles |
| **Time Efficiency** | `total_info_density / total_duration` | Agent respects time budget |
| **Credibility** | `avg(credibility_score)` of recommended set | Agent picks expert sources |
| **Redundancy** | Avg pairwise similarity within recommended set | Agent avoids repetition |

- [ ] `experiments/evaluate.py` — computes all metrics for both systems
- [ ] `experiments/results/` — save comparison tables + plots

### Step 4.3 — Generate Figures
- [ ] **Figure 1:** Architecture diagram (the Shield Model from Section 4)
- [ ] **Figure 2:** Spider/radar chart comparing baseline vs. agent across all metrics
- [ ] **Figure 3:** A/B dashboard screenshot showing a real example
- [ ] **Figure 4:** Manipulation score distribution: YouTube recommendations vs. Agent recommendations
- [ ] **Figure 5:** Case study — one query, full walkthrough of what was recommended and why

### Step 4.4 — Write the Paper (if targeting publication)

**Structure:**
1. **Introduction** — Frame via CHT / "Time Well Spent" movement. Cite iAgent as related work. State your unique contribution: algorithmic inversion.
2. **Related Work** — iAgent, manipulation-aware RS, DeArrow/EchoTube, LLM-agent RS surveys, beyond-engagement RecSys papers
3. **Methodology** — Modules 1–4, scoring function, optimization formulation
4. **Experimental Setup** — Datasets, queries, metrics, baseline
5. **Results** — Tables + figures from Step 4.2–4.3
6. **Discussion** — Regulatory relevance (EU DSA/AI Act), limitations, future work
7. **Conclusion**

**Target venues:** RecSys, AAAI, AAMAS, FAccT, BEYOND@RecSys workshop

---

## Quick Decision Checklist

Before you start coding, resolve these:

| # | Decision | Options | Recommendation |
|---|----------|---------|----------------|
| 1 | **Scope of v1** | Full system with all modules vs. MVP with core modules only | Start with Modules 1 + 3 only (manipulation + optimizer). Add Module 2 iteratively. |
| 2 | **Transcript dependency** | Require transcripts vs. work with title+description only | Title + description first; transcripts as enhancement. Many videos lack transcripts. |
| 3 | **UI framework** | HTML/CSS/JS vs. Streamlit | Streamlit for speed if this is for a paper demo; HTML/CSS/JS if you want a polished product. |
| 4 | **Tactic attribution** | Binary clickbait vs. multi-label tactics | Binary first, add tactics in v2 if time allows. |
| 5 | **LLM integration** | Use LLM for explanations vs. template-based | Templates for v1. LLM adds cost and latency. |
| 6 | **User study** | Automated metrics only vs. real user evaluation | Automated first. User study is a nice-to-have for the paper. |

---

## Timeline Summary

| Days | Phase | What You Ship |
|------|-------|---------------|
| **1–3** | Phase 0: Setup | Repo, env, API key, read iAgent |
| **4–10** | Phase 1: Data | Scraper, datasets, EDA, baseline |
| **11–16** | Phase 2a: Manipulation Detector | Clickbait classifier, emotion scorer, social proof stripper |
| **17–22** | Phase 2b: Quality Evaluator | Info density, credibility, goal alignment, bias |
| **23–24** | Phase 2c: Optimizer | Greedy submodular selection with time/diversity constraints |
| **25–31** | Phase 3: Integration | Full pipeline, FastAPI backend, A/B dashboard, explainability |
| **32–42** | Phase 4: Evaluation | Experiments, metrics, figures, paper draft |

> [!IMPORTANT]
> **Start with Phase 0 today.** The API key and repo setup are zero-risk tasks that unblock everything else. Reading the iAgent paper should be your evening task tonight.
