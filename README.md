# Sports Health & Injury Assistant 🏃‍♂️

A beginner-friendly public RAG application that answers educational questions about **sports injuries, athlete health, injury prevention, rehabilitation principles, and return to sport** using a curated knowledge base based on public sports-medicine sources.

**Live app:** https://suhail750-rag-engineering-lab-app-bfujle.streamlit.app/  
**Repository:** https://github.com/suhail750/rag-engineering-lab

## What visitors can ask

- Common sports injuries and warning signs.
- Ankle sprains, knee injuries, ACL concerns, muscle strains, tendon pain, and overuse injuries.
- Concussion recognition and safe return-to-sport guidance.
- Heat illness, hydration, and athlete safety.
- Injury prevention, rehabilitation principles, and return-to-sport decision-making.
- General questions about football, cricket, basketball, running, tennis, gym training, and other activities where the knowledge base contains relevant evidence.

The current knowledge pack is a starter educational library, not a complete clinical reference for every sport or condition. Answers are limited by the sources included in this repository.

## Safety boundary

**This is an educational information tool, not a medical device.** It cannot diagnose an injury from symptoms, prescribe medication or treatment, interpret scans, or clear an athlete to return to play. Similar symptoms can have different causes. Serious symptoms require medical assessment; suspected concussion and heat stroke need urgent attention. Always follow a qualified healthcare professional's advice and local emergency procedures.

## RAG features

- Shared, read-only curated knowledge base; no visitor PDF or personal-data upload.
- Evidence-grounded answers with inspectable retrieved passages and original source links.
- Dense, BM25, and Hybrid retrieval.
- Optional cross-encoder reranking.
- LangChain document splitting and LangChain + Groq generation.
- Built-in retrieval evaluation with Hit Rate@k, Recall@k, Precision@k, MRR@k, and nDCG@k.
- Extractive fallback when no Groq API key is configured.

## Knowledge sources

The initial pack is based on public guidance and educational material from CDC HEADS UP, CDC Heat and Athletes, AAOS OrthoInfo, the American Orthopaedic Society for Sports Medicine (AOSSM), NICE, and a team-physician return-to-sport consensus statement.

Original links are included in `sample_data/sources.json` and shown in the app. The notes are original summaries, not copied full-text pages.

## How the app works

```text
Public sports-medicine / public-health sources
                    ↓
          Curated educational notes
                    ↓
        LangChain text splitting
                    ↓
      Sentence Transformer embeddings
                    ↓
        Dense / BM25 / Hybrid search
                    ↓
      Optional cross-encoder reranking
                    ↓
          LangChain prompt → Groq
                    ↓
       Answer + evidence + source links
```

## LangChain usage

1. `raglab/ingestion.py` uses `Document` and `RecursiveCharacterTextSplitter`.
2. `raglab/generation.py` uses `ChatPromptTemplate | ChatGroq | StrOutputParser`.

The embedding and retrieval functions remain small Python functions so beginners can inspect the RAG pipeline.

## Run locally

Requires Python 3.10–3.12.

```bash
git clone https://github.com/suhail750/rag-engineering-lab.git
cd rag-engineering-lab
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Optional LLM answers require a Groq API key. Store it in Streamlit Community Cloud secrets or an environment variable; never commit it to GitHub. The default model is `openai/gpt-oss-120b`.

## Evaluation

Open the Evaluation tab and run the included sports-health retrieval questions. Metrics evaluate source retrieval, not clinical correctness or answer safety. High-impact answers still require expert review.

## Deployment

The app is deployed on Streamlit Community Cloud. The knowledge base is read-only and shared by all visitors; each visitor has a separate Streamlit session.

## Limitations and future work

- The corpus is curated and finite; it does not search the live web at question time.
- It is not a diagnostic or treatment tool.
- Coverage is not exhaustive and differs by sport and injury.
- Retrieval metrics do not prove clinical accuracy.
- A future version should undergo review by qualified sports-medicine professionals before any real-world health use.

## License

MIT. See LICENSE.
