# Agent-as-Shield Recommender System

Agent-as-Shield is a recommender system (RS) algorithm designed differently because AI agents are not human. Instead of optimizing for engagement metrics like click-through rates, watch time, and emotional appeal, it optimizes for utility, truthfulness, and goal-alignment.

## Differentiation from iAgent
While recent works like iAgent (ACL 2025) focus on instruction-tuned recommender models that guide users through interactive conversational search, Agent-as-Shield takes a fundamentally different approach: **algorithmic inversion**. Instead of acting as an interface to traditional engagement-driven platform results, our system acts as a protective layer (a "shield") that actively penalizes the very signals that human-centric RS models boost. We explicitly strip out social proof manipulations (views, likes), penalize emotional clickbait tactics, and re-rank strictly based on objective information density, goal alignment, and source credibility. This transforms the agent from a helpful navigator of an engagement-optimized ecosystem into a curator that forces the content ecosystem to adhere to rigorous agent-centric constraints.

## Project Structure
- `data/`: Raw and processed datasets, plus YouTube scrapes.
- `src/`: Core modules including scraper, manipulation detector, quality evaluator, optimizer, explainer, pipeline, and API.
- `notebooks/`: EDA and experiment notebooks.
- `experiments/`: Evaluation scripts and results.
- `frontend/`: Dashboard UI for A/B comparison.
