# NLP Coursework Projects

This repository gathers the Natural Language Processing coursework completed during the semester at the Asian Institute of Technology. It includes six major assignments, supporting web applications, trained model checkpoints, experiment outputs, demo media, and exam-preparation material.

## Repository Structure

- A1-Word2Vec/ - Word embedding assignment covering Word2Vec skip-gram, Word2Vec with negative sampling, GloVe from scratch, and GloVe with Gensim. Includes the research notebooks in A1/, a Django backend, a React frontend, pre-downloaded GloVe vectors in glove.6B/, and backup/output/trash folders used during experimentation.
- A2-LSTM_Language_Model/ - LSTM-based language modeling project with notebooks, a trained checkpoint in best-model.pt, vocabulary files, demo assets such as App.png and Working_App.mp4, and the full-stack app in lstm-language_model_app/.
- A3-Language_Translation/ - English-to-Nepali machine translation experiments with multiple attention variants and transformer-based seq2seq models. Includes notebooks, trained checkpoints under model/, generated outputs in result/, and the deployed application in translation-app/.
- A4-BERT_Sentence/ - BERT-based sentence modeling assignment with notebooks for a from-scratch BERT implementation and sentence-level fine-tuning, saved model assets, result screenshots, and the bert-sentence-app/ web interface.
- A5-Naive_RAG_vs_Contextual_Retrieval/ - Retrieval-augmented generation assignment comparing naive retrieval against contextual retrieval. Includes the main notebook, saved experiment artifacts in a5_outputs/, a web app in app/, and demo images and GIFs.
- A6-RAG_Techniques/ - Extended RAG techniques assignment with notebook experiments, evaluated outputs in answer/, supporting application code in app/, report material, and visual demo outputs.
- env/ - Shared Python virtual environment for the workspace.

## Folder Highlights

| Folder | Main Contents |
| --- | --- |
| A1-Word2Vec/ | A1 notebooks, backend/, frontend/, glove.6B/, requirements.txt, embedding/output files |
| A2-LSTM_Language_Model/ | A2.ipynb, LSTM LM.ipynb, best-model.pt, vocab.pkl, itos.pkl, lstm-language_model_app/ |
| A3-Language_Translation/ | A3.ipynb, A3_multiplicative_best.ipynb, model/, result/, translation-app/, demo media |
| A4-BERT_Sentence/ | A4.ipynb, A4_2.ipynb, model/, bert-sentence-app/, result screenshots and demos |
| A5-Naive_RAG_vs_Contextual_Retrieval/ | A5.ipynb, a5_outputs/, app/, A5_Output1.png, A5_Output2.png, a5.gif |
| A6-RAG_Techniques/ | A6.ipynb, answer/, app/, output1.png, output2.png, A6.gif, requirements.txt |

## Results and Artifacts

The repository already contains the main result files produced across the assignments:

- A1-Word2Vec/ - Trained embedding artifacts such as skipgram_neg_embeddings.json, experiment data files, and generated outputs under A1/output/ and backend/output/.
- A2-LSTM_Language_Model/ - Trained checkpoint best-model.pt, vocabulary mappings, application screenshot App.png, and demo video Working_App.mp4.
- A3-Language_Translation/ - Multiple saved translation checkpoints in model/, generated translation results in result/, and demo assets including Best_result.mp4, best_result_1.png, best_result_2.png, and GIF recordings.
- A4-BERT_Sentence/ - Model files in model/ with result visuals including Result-Mask.png, Result-Similarity.png, BERT.gif, and BERT.mp4.
- A5-Naive_RAG_vs_Contextual_Retrieval/ - Evaluation and training artifacts inside a5_outputs/ along with A5_Output1.png, A5_Output2.png, and a5.gif.
- A6-RAG_Techniques/ - Final assignment answers in answer/, visual outputs output1.png and output2.png, and A6.gif showing the application/demo flow.
- Most assignment folders also include PDF reports or assignment briefs documenting the task requirements and submission context.

## Getting Started

1. Activate the shared virtual environment from env/ or the assignment-specific environment when one is provided.
2. Install Python dependencies from the local requirements.txt file inside the assignment you want to run.
3. Open the relevant notebook in VS Code or Jupyter to reproduce training, evaluation, and analysis.
4. For folders that contain backend/frontend or app/ projects, start the backend first and then launch the frontend.

## Notes

- Several assignments include full-stack demo applications in addition to the notebooks.
- Backup and trash folders are preserved in some projects to retain earlier notebook versions and intermediate work.
- Large pretrained or trained model artifacts are stored directly in the repository for coursework demonstration.

---

**Author**:
Rahul Shakya - st125982 <br />
Student Email: st125982@ait.asia <br/>
Personal Email: shakyarahul435@gmail.com <br />
University: <a href="https://ait.ac.th/"><b>Asian Institute of Technology, Thailand</b></a>
