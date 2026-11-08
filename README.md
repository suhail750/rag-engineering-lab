# Sports Health & Athlete Safety RAG 🏃‍♂️

<https://suhail750-rag-engineering-lab-app-bfujle.streamlit.app/>

🚀 **[OPEN THE LIVE SPORTS RAG APP](https://suhail750-rag-engineering-lab-app-bfujle.streamlit.app/)**

A public Retrieval-Augmented Generation (RAG) app focused on **sports injuries, athlete safety, injury prevention, rehabilitation principles, and return to sport**. It retrieves evidence from a curated knowledge pack based on public sports-medicine and public-health sources, then uses LangChain and Google Gemini to explain the evidence in plain language.

**Live app:** https://suhail750-rag-engineering-lab-app-bfujle.streamlit.app/  
**Repository:** https://github.com/suhail750/rag-engineering-lab

## What makes this project different?

This is not a generic RAG chatbot. Its knowledge base is dedicated to sports health and athlete safety. It is designed to help athletes, coaches, and learners understand reliable general information and find the original source material.

## What can visitors ask?

- Common sports injuries and urgent warning signs.
- Ankle sprains, knee injuries, ACL concerns, and safe general recovery principles.
- Muscle strains, tendon pain, and overuse injuries.
- Concussion recognition and return-to-sport safety.
- Heat illness, hydration, and athlete safety.
- Injury prevention, rehabilitation principles, and return-to-sport decision-making.

The current knowledge pack is a starter educational library, not a complete clinical reference for every sport, injury, or athlete.

## Safety boundary

**This is an educational information tool, not a medical device.** It cannot diagnose injuries, prescribe medication or treatment, interpret scans, or clear anyone to return to play. Similar symptoms can have different causes. Suspected concussion and heat stroke need urgent attention. Follow qualified healthcare professionals and local emergency procedures.

## RAG features

- Shared, read-only curated sports-health knowledge base.
- Evidence-grounded answers with inspectable retrieved passages and original source links.
- Dense, BM25, and Hybrid retrieval.
- Optional cross-encoder reranking.
- LangChain document splitting and LangChain + Gemini generation.
- Built-in retrieval evaluation: Hit Rate@k, Recall@k, Precision@k, MRR@k, and nDCG@k.
- Extractive fallback when no Gemini API key is configured.

## Knowledge sources

The initial pack contains original educational summaries based on:
- American Orthopaedic Society for Sports Medicine (AOSSM).
- AAOS OrthoInfo.
- US Centers for Disease Control and Prevention (CDC HEADS UP and Heat and Athletes).
- NICE rehabilitation guidance.
- Team Physician Consensus Statement on Return to Sport/Return to Play.

Original links are stored in `sample_data/sources.json` and shown in the app. The knowledge notes are concise summaries, not copied full-text webpages.

## How the RAG pipeline works

```text
Public sports-medicine and public-health sources
                    ↓
          Curated sports-health notes
                    ↓
        LangChain text splitting
                    ↓
      Sentence Transformer embeddings
                    ↓
        Dense / BM25 / Hybrid search
                    ↓
      Optional cross-encoder reranking
                    ↓
          LangChain prompt → Gemini
                    ↓
       Answer + retrieved evidence + source links
```

## LangChain usage

1. `raglab/ingestion.py` uses LangChain `Document` and `RecursiveCharacterTextSplitter`.
2. `raglab/generation.py` uses `ChatPromptTemplate | ChatGemini | StrOutputParser`.

The embedding and retrieval functions remain small Python functions so beginners can inspect the pipeline.

## Run locally

Requires Python 3.10–3.12.

```bash
git clone https://github.com/suhail750/rag-engineering-lab.git
cd rag-engineering-lab
python -m venv .venv
pip install -r requirements.txt
streamlit run app.py
```

Create a free Gemini API key in [Google AI Studio](https://aistudio.google.com/app/apikey), then add `GEMINI_API_KEY = "your-key"` under Streamlit Community Cloud → App settings → Secrets. You can also set `GEMINI_API_KEY` as a local environment variable. Never commit the key to GitHub. Without it, the app displays retrieved evidence only.

## Evaluation

Open the Evaluation tab and run the sports-health retrieval questions in `sample_data/evaluation.json`. These metrics test retrieval against labelled source notes; they do not prove medical correctness or answer safety.

## Deployment and limitations

The app is designed for Streamlit Community Cloud. The knowledge base is read-only and shared by visitors; each visitor has a separate session. The current corpus is curated and finite; it does not search the live web at question time. Coverage is not exhaustive and should be reviewed by qualified sports-medicine professionals before use in high-stakes settings.

## License

MIT. See LICENSE.
