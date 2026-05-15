# University-Level Source Documents

This folder contains the PDF source documents used for the university-level mathematics problems.

At this stage, these PDF files are provided as source/problem documents. A structured CSV/JSON version for automated evaluation is not currently included.

The university-level evaluation script expects a future structured file such as:

`university_level_subquestions.csv`

Required columns:

- `qid`
- `subid`
- `subject`
- `full_question`
- `subquestion`

Therefore, the university-level evaluation script should currently be treated as a reference/work-in-progress pipeline rather than a fully runnable component.
