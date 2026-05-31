# Scientific Workflow Prompt Pack

Prepared for: **research assistants**  
Default style: **concise and precise**

> 1 high-priority task should be reviewed first.

## Task Overview

| # | Task | Domain | Priority | Output |
|---|------|--------|----------|--------|
| 1 | Extract ALD process parameters | materials science | HIGH | json |
| 2 | Summarize survey table findings | psychology | MEDIUM | markdown |
| 3 | Check ontology alignment candidates | knowledge engineering | LOW | csv |

## Details

### 1. Extract ALD process parameters

**Goal:** Extract process temperature, precursor names, pulse time, purge time, and substrate from a methods section.

**Recommended prompt file:** `outputs/prompt_extract-ald-process-parameters.txt`

**Evidence requirement:** Ask the model to preserve source snippets or citation anchors.

**Context files:**
- `paper_methods.txt`
- `schema_v1.json`

**Expected fields:** material, precursor, temperature_c, pulse_time_s, purge_time_s, substrate

### 2. Summarize survey table findings

**Goal:** Summarize which constructs and instruments are reported in a survey table.

**Recommended prompt file:** `outputs/prompt_summarize-survey-table-findings.txt`

**Evidence requirement:** No citation anchors required for this demo task.

**Context files:**
- `survey_table.csv`

**Expected fields:** construct, instrument, population, sample_size

### 3. Check ontology alignment candidates

**Goal:** Review candidate mappings between two ontologies and flag uncertain matches.

**Recommended prompt file:** `outputs/prompt_check-ontology-alignment-candidates.txt`

**Evidence requirement:** No citation anchors required for this demo task.

**Context files:** None provided. The prompt should ask the model to work only with the supplied mapping rows.

**Expected fields:** source_class, target_class, relation, confidence, comment

