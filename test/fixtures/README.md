# Real-model fixture

`ch_en_num.expected.json` records the output from RapidOCR v3.10.0 with the
official PP-OCRv6 small detector and recognizer, the PP-OCRv4 mobile direction
classifier, and the matching PP-OCRv6 dictionary. `pnpm test:e2e` downloads
and verifies those pinned assets and the upstream `ch_en_num.jpg`,
`ch_doc_server.png`, `text_cls.jpg`, and `text_rec.jpg` test images into the
ignored `models/` directory automatically.

`ch_en_num.expected.json` and `ch_doc_server.expected.json` record upstream
full-page outputs. `ch_en_num_162x215.expected.json` records the upstream output
for a 162 × 215 JPEG derived from `ch_en_num.jpg` during the test. The source
images remain hosted by the RapidOCR repositories and are not redistributed
here. Model sources and checksums are listed in `model-assets.mjs` and
`MODEL-LICENSES.md`.
