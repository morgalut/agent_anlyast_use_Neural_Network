"""
nn_promat_core.py
═══════════════════════════════════════════════════════════════════════════════
Neural-Network PROMAT Core — shared instruction set for all ORC pipeline agents.

Replaces the old additive weight system (+20, +15, -50 …) with a deterministic
6-layer neural architecture.  Every decision is driven by signal activations,
pattern logic, a cross-sheet dependency graph, hard firewalls, a normalised
confidence distribution, and a final business arbitration layer.

No numeric scores.  No keyword bonuses.  No additive penalties.
Only binary signals → pattern logic → graph analysis → firewalls →
softmax → business arbitration.

Layer summary
─────────────
  L0  Input extraction    — raw OCR, no interpretation
  L1  Binary activations  — each signal is 0 or 1, never partial
  L2  Pattern logic       — AND / OR / NOT combinations of L1 signals
  L3  Context graph       — cross-sheet dependency analysis
  L4  Hard gates          — firewalls that permanently block a sheet
  L5  Confidence softmax  — normalised probability over passing sheets
  L6  Business arbitration — bridges technical correctness to business correctness
═══════════════════════════════════════════════════════════════════════════════
"""

# ──────────────────────────────────────────────────────────────────────────────
#  LAYER 0 — Input extraction
# ──────────────────────────────────────────────────────────────────────────────

NN_LAYER_0 = """
╔══════════════════════════════════════════════════════════╗
║  LAYER 0 — Input Extraction                             ║
╚══════════════════════════════════════════════════════════╝

Extract the following raw features for EVERY sheet before any reasoning.
Do NOT interpret, score, or rank at this stage — extract only.

  F0  sheet_name          String name as read from the workbook
  F1  is_hidden           Boolean — is the sheet hidden or veryHidden?
  F2  headers             List of non-empty values from row 1
  F3  sample_rows         Rows 2–6 as raw values
  F4  formula_samples     Up to 30 cell values that start with "="
  F5  formula_sheet_refs  Sheet names extracted from formula strings
                          (e.g. '=SUMIF(SAP!I:I,…)' → ["SAP"])
  F6  all_values_flat     Up to 300 unique string values from the sheet body
  F7  is_active_sheet     Boolean — is this the workbook's active sheet?

Extraction rule: open the workbook with data_only=False to read formula
strings, not computed results.  Formulas are the primary evidence.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  LAYER 1 — Binary activations
# ──────────────────────────────────────────────────────────────────────────────

NN_LAYER_1 = """
╔══════════════════════════════════════════════════════════╗
║  LAYER 1 — Binary Signal Activations                    ║
╚══════════════════════════════════════════════════════════╝

For each signal below, compute exactly 0 (off) or 1 (on).
There is NO partial activation.  No "almost 1".  Only 0 or 1.

COA_SIGNAL
  ON if all_values_flat contains at least 3 of these strings
  (case-insensitive):
    "assets" | "current assets" | "long-term assets" |
    "liabilities and equity" | "current liabilities" |
    "long-term liabilities" | "equity"

FORMULA_SIGNAL
  ON if formula_samples is non-empty (≥ 1 formula exists)

CROSS_REF_SIGNAL
  ON if formula_sheet_refs is non-empty
  (this sheet's formulas reference at least one other sheet)

REFERENCED_BY_SIGNAL
  ON if OTHER sheets contain formulas that reference THIS sheet
  (computed in Layer 3 after cross-sheet graph is built)

COMPANY_COLUMN_SIGNAL
  ON if headers contains ≥ 1 value that includes any of:
    "NIS" | "$" | "DOLLAR" | "USD" | "ILS" | "INC" | "LTD"

AJE_SIGNAL
  ON if headers contains a value that includes "AJE" or "ADJUSTING"

CONSOLIDATE_SIGNAL
  ON if headers contains a value that includes
    "CONSOLIDATE" | "CONSOL" | "TOTAL"

CODE_COLUMN_SIGNAL
  ON if there exists a column whose non-empty values ALL match
  the regular expression ^[A-Za-z0-9]+$ (alphanumeric only, no spaces)
  AND COA_SIGNAL is OFF
  (this is the account-code column characteristic of a TB / ledger sheet)

FINAL_COLUMN_SIGNAL
  ON if headers contains a value that includes
    "FINAL" | "TRIAL BALANCE" | " TB "

HIDDEN_SIGNAL
  ON if is_hidden = true

Layer 1 rule: every signal is binary.  If you are uncertain, default to 0.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  LAYER 2 — Pattern logic
# ──────────────────────────────────────────────────────────────────────────────

NN_LAYER_2 = """
╔══════════════════════════════════════════════════════════╗
║  LAYER 2 — Pattern Logic (Hidden Layer 1)               ║
╚══════════════════════════════════════════════════════════╝

Combine Layer-1 signals into three composite patterns.

FS_PATTERN  (Full Financial Statement — main sheet candidate)
  = COA_SIGNAL
    AND FORMULA_SIGNAL
    AND CROSS_REF_SIGNAL
    AND COMPANY_COLUMN_SIGNAL
    AND NOT HIDDEN_SIGNAL
    AND NOT CODE_COLUMN_SIGNAL

  Interpretation: this sheet is a reporting / output sheet that
  aggregates data from a TB sheet via formulas.
  ALL six conditions must be satisfied simultaneously.

TB_PATTERN  (Trial Balance / Ledger sheet)
  = CODE_COLUMN_SIGNAL
    AND FINAL_COLUMN_SIGNAL
    AND NOT COA_SIGNAL
    AND NOT COMPANY_COLUMN_SIGNAL

  Interpretation: this is a data-source sheet, not a reporting sheet.
  A sheet matching TB_PATTERN cannot be the main sheet.

PARTIAL_FS_PATTERN  (Partial evidence — needs deeper verification)
  = COA_SIGNAL
    AND (FORMULA_SIGNAL OR CROSS_REF_SIGNAL)
    AND NOT HIDDEN_SIGNAL
    AND NOT TB_PATTERN

  Interpretation: structural evidence exists but is incomplete.
  Requires Layer-3 graph confirmation before promotion.

Priority rules:
  FS_PATTERN overrides PARTIAL_FS_PATTERN.
  TB_PATTERN disqualifies a sheet from main-sheet candidacy (triggers GATE_3).
  A sheet can match both TB_PATTERN and PARTIAL_FS_PATTERN simultaneously —
  in that case TB_PATTERN wins and the sheet is disqualified.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  LAYER 3 — Context graph  (cross-sheet attention)
# ──────────────────────────────────────────────────────────────────────────────

NN_LAYER_3 = """
╔══════════════════════════════════════════════════════════╗
║  LAYER 3 — Context Graph (Hidden Layer 2)               ║
╚══════════════════════════════════════════════════════════╝

Build a directed dependency graph across ALL sheets.

Graph construction:
  For every sheet X: edges X → Y for each Y in X.formula_sheet_refs.
  After building: for every sheet Y, collect all sheets X that point to it
  (incoming_refs of Y).

Attention rules:

[A1 — Directional identity]
  A sheet with outgoing edges (points to others) = FS candidate.
  A sheet with incoming edges (pointed to by others) = TB / source candidate.
  If BOTH: check direction.  FS → TB is valid.  TB → FS is invalid.

[A2 — Reference concentration]
  The sheet most frequently referenced by others = primary TB / source sheet.
    → assign role_in_graph = "TB"
  The sheet with the most outgoing references = primary FS candidate.
    → assign role_in_graph = "FS"

[A3 — Intermediate sheet detection]
  If X → Y → Z:
    X = main sheet (FS)
    Y = intermediate sheet
    Z = TB source
  Document this chain explicitly.

[A4 — Single-company-column CONSOLIDATE check]
  If COMPANY_COLUMN_SIGNAL = 1 but only ONE company column exists:
    Check if an intermediate sheet (A3) contains COA structure AND
    multiple company columns.
    If YES → current sheet is a CONSOLIDATE sheet (still valid main sheet).
    Set consolidate = true.
    This sheet is EXEMPT from GATE_2 (see Layer 4).

[A5 — Active sheet boost]
  If is_active_sheet = true AND FS_PATTERN = 1 → set attention_boost = true.
  This increases confidence slightly in Layer 5 but does NOT bypass gates.

Layer 3 output per sheet:
  {
    "outgoing_refs":  [...],   // sheets this sheet references
    "incoming_refs":  [...],   // sheets that reference this sheet
    "role_in_graph":  "FS" | "TB" | "INTERMEDIATE" | "UNKNOWN",
    "consolidate":    true/false,
    "attention_boost": true/false
  }
"""

# ──────────────────────────────────────────────────────────────────────────────
#  LAYER 4 — Hard gates  (firewalls)
# ──────────────────────────────────────────────────────────────────────────────

NN_LAYER_4 = """
╔══════════════════════════════════════════════════════════╗
║  LAYER 4 — Hard Gates (Firewalls)                       ║
╚══════════════════════════════════════════════════════════╝

Hard gates permanently block a sheet from being the main sheet.
They are NOT penalties.  They are logical firewalls.
A blocked sheet receives confidence = 0 in Layer 5.
There is NO compensating evidence that overrides a fired gate.
A sheet that triggers any gate is DISQUALIFIED — final, irreversible.

NOTE: Layer 4 disqualifies sheets from the TECHNICAL path only.
Layer 6 Business Arbitration may later reconsider sheets blocked by GATE_2
if the block is classified as TECHNICAL (not CRITICAL).
GATE_1, GATE_3, and GATE_4 are ALWAYS critical and can never be overridden.

GATE_1  [HIDDEN FIREWALL]
  Condition: HIDDEN_SIGNAL = 1
  Action:    BLOCK — a hidden sheet can NEVER be the main sheet.
  Disqualification class: CRITICAL — not overrideable by Layer 6.

GATE_2  [NO COMPANY COLUMNS FIREWALL]
  Condition: COMPANY_COLUMN_SIGNAL = 0
             AND CONSOLIDATE_SIGNAL = 0
             AND consolidate (Layer 3, A4) = false
  Action:    BLOCK — without company columns, this is not a FS sheet
             under the technical definition.
  Disqualification class: TECHNICAL — Layer 6 may override for REPORTING_FS
             sheets that are semantically presentation-final.
  Exemption: sheets confirmed as CONSOLIDATE by Layer 3 A4 are exempt.

  CRITICAL NOTE: This gate is the primary defence against CF-type errors.
  A sheet named "CF" or "Cash Flow" that lacks company columns MUST be
  blocked here, regardless of how many keywords match its name.

GATE_3  [TRIAL BALANCE FIREWALL]
  Condition: TB_PATTERN = 1 (Layer 2)
  Action:    BLOCK — a TB / ledger sheet is a source, not an output.
  Disqualification class: CRITICAL — not overrideable by Layer 6.

GATE_4  [INCOMING-ONLY FIREWALL]
  Condition: role_in_graph = "TB" (Layer 3)
             AND FS_PATTERN = 0
             AND PARTIAL_FS_PATTERN = 0
  Action:    BLOCK — a pure source sheet without any FS signal is not main.
  Disqualification class: CRITICAL — not overrideable by Layer 6.

Gate evaluation order: GATE_1 → GATE_2 → GATE_3 → GATE_4.
Stop at the first triggered gate; do not evaluate further gates.

Layer 4 output per sheet:
  { "passed": true/false, "blocked_by": "GATE_1" | "GATE_2" | "GATE_3" | "GATE_4" | null }
"""

# ──────────────────────────────────────────────────────────────────────────────
#  LAYER 5 — Confidence softmax
# ──────────────────────────────────────────────────────────────────────────────

NN_LAYER_5 = """
╔══════════════════════════════════════════════════════════╗
║  LAYER 5 — Confidence Softmax (Output Layer)            ║
╚══════════════════════════════════════════════════════════╝

Only sheets that passed Layer 4 (passed = true) participate.
Blocked sheets have confidence = 0.0 and are excluded.

Signal strength per passing sheet:
  S(sheet) = (
    FS_PATTERN        × 1.00  +   // full FS structure
    PARTIAL_FS_PATTERN × 0.50  +   // partial structure
    AJE_SIGNAL         × 0.20  +   // AJE column reinforces FS identity
    CONSOLIDATE_SIGNAL × 0.15  +   // CONSOLIDATE column reinforces
    attention_boost    × 0.15  +   // active sheet + FS_PATTERN boost
    CROSS_REF_SIGNAL   × 0.10      // outgoing formula references
  )

Softmax normalisation:
  confidence(sheet_i) = S(sheet_i) / Σ S(all_passing_sheets)

  Result: confidence values sum to 1.0 across all passing sheets.
  A sheet that is the ONLY passer receives confidence = 1.0.

Decision thresholds:
  confidence ≥ 0.70  →  CONFIRMED main sheet   (main_sheet_confirmed = true)
  confidence ≥ 0.40  →  POSSIBLE main sheet    (report uncertainty)
  confidence < 0.40  →  NOT FOUND              (main_sheet_exists = false)

Tie-breaking (two sheets within 0.10 of each other):
  Step 1: Apply Layer-3 A1 — which sheet points to the other?
          The pointing sheet = FS.  The pointed-to sheet = TB.
  Step 2: If still tied, choose the sheet with more COA_SIGNAL activations
          (count how many of the 7 required sections are present).
  Step 3: If still tied, flag for manual review.

Layer 5 produces the TECHNICAL winner.
Layer 6 then determines whether the business-correct winner differs.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  LAYER 6 — Business arbitration  (semantic bridge)
# ──────────────────────────────────────────────────────────────────────────────

NN_LAYER_6 = """
╔══════════════════════════════════════════════════════════╗
║  LAYER 6 — Business Arbitration (Semantic Bridge)       ║
╚══════════════════════════════════════════════════════════╝

Purpose:
  Layer 6 resolves the gap between:
    • technically correct main sheet  (L5 softmax winner)
    • business-correct final reporting sheet  (what a finance user expects)

  Layers 1–5 identify the strongest TECHNICAL winner.
  Layer 6 decides whether that technical winner also matches the sheet a
  finance user would call the main output.

Key principle:
  Do NOT weaken Layer-4 gates globally.
  Instead, distinguish between:
    • CRITICAL disqualification   → can NEVER be final main sheet
    • TECHNICAL disqualification  → may still be business-correct

The core insight:
  A sheet that is the structural / computational hub of a workbook
  (many outgoing refs, company columns, AJE signals) is not necessarily
  the FINAL REPORTING OUTPUT a user cares about.
  AJE = adjustment staging hub   ≠   Report = final presentation output.
  Layer 6 resolves this conflict.

──────────────────────────────────────────────────────────
BUSINESS SIGNALS
──────────────────────────────────────────────────────────

CANONICAL_FS_TITLE_SIGNAL
  ON if sheet_name or dominant title contains any of (case-insensitive):
    "balance sheet" | "balance sheets" |
    "statement of operations" | "statement of income" |
    "profit and loss" | "p&l" |
    "cash flow" | "cash flows" | "statement of cash flows" |
    "change in equity" | "stockholders' equity" |
    "financial statements" | "report"

PRESENTATION_LAYOUT_SIGNAL
  ON if the sheet appears to be a human-facing financial report rather than
  a staging / journal / source sheet.
  Typical evidence:
    • readable financial line items
    • period / year labels
    • statement-style sections
    • no raw journal-entry structure
    • no source-ledger code-column dominance

STAGING_SHEET_SIGNAL
  ON if sheet_name, headers, or dominant structure indicates:
    "aje" | "adjusting" | "adjustments" | "elimination" |
    "mapping" | "bridge" | "rollforward" | "support" | "schedule"

FINAL_OUTPUT_ROLE_SIGNAL
  ON if the sheet is likely the human-facing final output sheet.
  Typical evidence:
    • canonical FS title
    • presentation layout
    • report-style formatting / headings
    • appears downstream from adjustment / source sheets

──────────────────────────────────────────────────────────
SHEET TYPE CLASSIFIER
──────────────────────────────────────────────────────────

Assign exactly one sheet_type per sheet.
Apply rules in the following priority order — first match wins:

  SOURCE_TB
    If TB_PATTERN = 1
    OR role_in_graph = "TB" and source-sheet structure dominates

  REPORTING_FS (when FS_PATTERN = 1)
    If FS_PATTERN = 1 AND sheet name does NOT explicitly mark it as
    a staging layer (i.e., name does not contain "aje", "adjusting",
    "adjustments", "elimination").
    IMPORTANT: FS_PATTERN = 1 is checked BEFORE AJE_SIGNAL.
    A consolidation FS sheet may contain an "AJE" adjustment column
    internally — that does NOT make the sheet itself a staging layer.
    Only a sheet explicitly named as staging (e.g., named "AJE") is
    demoted to ADJUSTMENT_STAGING when FS_PATTERN = 1.

  ADJUSTMENT_STAGING (only when FS_PATTERN = 0)
    If AJE_SIGNAL = 1
    OR STAGING_SHEET_SIGNAL = 1
    OR the sheet name/title contains "aje", "adjusting", "adjustments",
       "elimination", "mapping", "bridge", "rollforward", "support",
       "schedule"

  REPORTING_FS (when FS_PATTERN = 0 but strong title evidence)
    If sheet name or title contains a STRONG canonical FS keyword:
      "balance sheet" | "balance sheets" |
      "statement of operations" | "statement of income" |
      "profit and loss" | "p&l" |
      "cash flow" | "cash flows" | "statement of cash flows" |
      "change in equity" | "stockholders' equity" | "financial statements"
    OR contains the WEAK keyword "report" AND COA_SIGNAL = 1.
    NOTE: "report" alone (without COA evidence) is insufficient —
    it is too common in dashboard and summary sheets.

  INTERMEDIATE_CONSOLIDATION
    If consolidate = true
    OR the sheet sits in FS → INTERMEDIATE → TB chains

  AUXILIARY_SCHEDULE
    If COA_SIGNAL = 1 but CROSS_REF_SIGNAL = 0 (no outgoing references)

  UNKNOWN
    Otherwise

──────────────────────────────────────────────────────────
DISQUALIFICATION CLASS
──────────────────────────────────────────────────────────

Assign exactly one disqualification class per sheet:

  CRITICAL
    If blocked_by ∈ { GATE_1, GATE_3, GATE_4 }
    These sheets can NEVER be final main sheet — no override possible.

  TECHNICAL
    If blocked_by = GATE_2 only
    AND the sheet shows strong REPORTING_FS evidence.
    Example: canonical report sheet missing company columns.
    Layer 6 may override this block when other semantic conditions hold.

  NONE
    If layer4.passed = true (no gate fired)

──────────────────────────────────────────────────────────
BUSINESS ARBITRATION RULES
──────────────────────────────────────────────────────────

BA1 — Technical winner
  The highest-confidence passing sheet from Layer 5 is the technical winner.

BA2 — Business winner candidate
  Search all visible sheets for the strongest REPORTING_FS candidate.

BA3 — Safe override condition
  Override the technical winner with the business winner ONLY if ALL
  of the following are true:

    1. technical_winner.sheet_type is "ADJUSTMENT_STAGING"
       OR "INTERMEDIATE_CONSOLIDATION"

    2. business_candidate.sheet_type = "REPORTING_FS"

    3. business_candidate.disqualification_class = "TECHNICAL"
       OR business_candidate.disqualification_class = "NONE"

    4. business_candidate is NOT blocked by GATE_1, GATE_3, or GATE_4

    5. business_candidate shows FINAL_OUTPUT_ROLE_SIGNAL = 1
       (is clearly the human-facing FS output)

BA4 — No unsafe override
  Never override to a sheet that is:
    • SOURCE_TB
    • hidden (GATE_1)
    • trial balance (GATE_3)
    • pure incoming-only source (GATE_4)

BA5 — Dual truth preservation
  If BA3 conditions are met (override applies):
    technical_main_sheet = technical winner    (preserved for traceability)
    business_main_sheet  = reporting candidate
    final_main_sheet     = business_main_sheet
    decision_mode        = "business_override"

  Otherwise (no override):
    technical_main_sheet = technical winner
    business_main_sheet  = technical winner
    final_main_sheet     = technical winner
    decision_mode        = "technical_default"

──────────────────────────────────────────────────────────
LAYER 6 OUTPUT
──────────────────────────────────────────────────────────

  {
    "technical_main_sheet":  "<sheet>" | null,
    "business_main_sheet":   "<sheet>" | null,
    "final_main_sheet":      "<sheet>" | null,
    "decision_mode":         "technical_default" | "business_override" | "no_valid_sheet",
    "sheet_types": {
      "<sheet>": "REPORTING_FS|ADJUSTMENT_STAGING|SOURCE_TB|INTERMEDIATE_CONSOLIDATION|AUXILIARY_SCHEDULE|UNKNOWN"
    },
    "disqualification_classes": {
      "<sheet>": "CRITICAL|TECHNICAL|NONE"
    },
    "business_arbitration": {
      "technical_winner_sheet_type": "...",
      "business_candidate": "<sheet>" | null,
      "business_candidate_sheet_type": "...",
      "business_candidate_blocked_by": "<gate>" | null,
      "business_candidate_disqualification_class": "CRITICAL|TECHNICAL|NONE|null",
      "override_applied": true/false
    }
  }
"""

# ──────────────────────────────────────────────────────────────────────────────
#  DECISION PROTOCOL
# ──────────────────────────────────────────────────────────────────────────────

NN_DECISION_PROTOCOL = """
╔══════════════════════════════════════════════════════════╗
║  DECISION PROTOCOL — Full 6-Layer Execution             ║
╚══════════════════════════════════════════════════════════╝

Execute these steps in strict order for every sheet in the workbook:

  Step 1 — L0: Extract F0–F7 for every sheet (OCR, no interpretation).
  Step 2 — L1: Compute all 9 binary signals per sheet.
  Step 3 — L2: Compute FS_PATTERN, TB_PATTERN, PARTIAL_FS_PATTERN per sheet.
  Step 4 — L3: Build the cross-sheet dependency graph.
               Assign role_in_graph to every sheet.
               Detect CONSOLIDATE (A4) and intermediate sheets (A3).
  Step 5 — L4: Apply GATE_1 through GATE_4.
               Any triggered gate → permanent technical disqualification.
               Record blocked_by for each disqualified sheet.
  Step 6 — L5: Compute S(sheet) and softmax confidence for passing sheets.
               Identify the TECHNICAL winner.
  Step 7 — L6: Run business arbitration.
               Classify every sheet into a sheet_type.
               Classify every blocked sheet into a disqualification_class.
               If the technical winner is ADJUSTMENT_STAGING or
               INTERMEDIATE_CONSOLIDATION, search for a REPORTING_FS
               candidate blocked only by a TECHNICAL gate.
               If safe override conditions (BA3) hold → apply override.
  Step 8 — Decision: emit final_main_sheet, technical_main_sheet,
               business_main_sheet, decision_mode.

Output contract:
  {
    "main_sheet_exists":    true/false,
    "main_sheet_name":      "<final sheet name>" | null,
    "technical_main_sheet": "<sheet name>" | null,
    "business_main_sheet":  "<sheet name>" | null,
    "decision_mode":        "technical_default" | "business_override" | "no_valid_sheet",
    "confidence":           0.0 – 1.0,
    "runner_up":            "<sheet name>" | null,
    "main_source_sheet":    "<TB sheet name>" | null,
    "nn_layers":            { ... full layer evidence including L6 ... }
  }

Detector override rule:
  The heuristic detector result is a CANDIDATE HINT only.
  It must pass all NN layers independently.
  If the detector's candidate fails any gate → it is blocked,
  and the NN winner is returned instead.
  Never return the detector result without NN validation.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  FULL PROMAT BLOCK  (imported by all agent files)
# ──────────────────────────────────────────────────────────────────────────────

FULL_NN_PROMAT = f"""
{'═' * 62}
  NEURAL PROMAT SYSTEM  v3.0
  6-layer neural architecture with business arbitration.
  No weights.  No bonuses.  No penalties.
  Binary signals → pattern logic → graph → firewalls →
  softmax → business arbitration.
{'═' * 62}

{NN_LAYER_0}

{NN_LAYER_1}

{NN_LAYER_2}

{NN_LAYER_3}

{NN_LAYER_4}

{NN_LAYER_5}

{NN_LAYER_6}

{NN_DECISION_PROTOCOL}
"""