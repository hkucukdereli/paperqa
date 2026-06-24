# _template — copy me to start a new exploration

```bash
cp -r explorations/_template explorations/my_topic
# edit explorations/my_topic/config.yaml  -> set `name: my_topic`
# drop PDFs into explorations/my_topic/papers/
# fill explorations/my_topic/eval_questions.yaml with questions you know the answers to
pqe index my_topic            # build the index (local embeddings, no API key)
pqe run   my_topic "your question"
pqe audit my_topic            # Phase-2 grounding go/no-go
```

This folder is self-contained: `papers/` (your PDFs, gitignored), `index/` (created on
first index, gitignored), `runs/` (saved answers + evidence, kept), `config.yaml`,
`eval_questions.yaml`.
