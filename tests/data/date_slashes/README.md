These are date-field crops from the locally supplied NIST SD19 sample forms.
`f0128_00.png` contains the handwritten date `8/3/89`, with a single-digit day
and month. The crop is copied without modifying its handwriting.

Tests cover parsing, image segmentation, and OCR retry decisions without loading
models. Actual OCR is verified separately using the locally cached model.
