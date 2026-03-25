
# ──────────────────────────────────────────────────────────────────────────────
#  F0 — Sheet-name fidelity
# ──────────────────────────────────────────────────────────────────────────────

NN_SHEET_NAME_FIDELITY = """
╔══════════════════════════════════════════════════════════╗
║  F0 — SHEET-NAME FIDELITY RULE                          ║
╚══════════════════════════════════════════════════════════╝

All sheet reasoning must operate on the canonical workbook sheet-name universe
discovered in Layer 0.

NON-NEGOTIABLE RULES
  • A valid sheet name is an exact worksheet tab name from the workbook.
  • Titles, captions, headings, report labels, and semantic business concepts
    are NOT sheet names.
  • Never invent aliases, normalized names, umbrella labels, or semantic parent
    entities such as "External Reporting Scheme" unless that exact text is a
    real workbook tab name.
  • Every returned sheet-bearing field must use exact workbook tab names only.

This applies to:
  • main_sheet_name
  • technical_main_sheet
  • presentation_main_sheet
  • business_main_sheet
  • technical_tb_sheet
  • is_card_sheet
  • main_source_sheet_name
  • strongest_candidate
  • runner_up
  • verification_target
  • fallback_candidate
  • header_sheets
  • relationship.main_to_tb_path
  • all sheet_evidence keys
  • all graph references such as incoming_refs / outgoing_refs / path_to_tb

Examples:
  • If the tab name is `P&L` and the visible title says
    "CONSOLIDATED STATEMENTS OF OPERATIONS", return `P&L`.
  • If the real workbook output is spread across `BS` and `P&L`, do not invent
    a synthetic parent reporting node.
  • If a business/reporting concept cannot be mapped to an exact tab name, do
    not emit it as a sheet name.
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

Also build the canonical workbook sheet-name universe.
This exact set is the only allowed sheet-name set for all later layers.
Any later candidate, evidence key, graph node, or path node not in this set is invalid.

  F0  sheet_name                String name as read from the workbook
  F1  is_hidden                 Boolean — is the sheet hidden or veryHidden?
  F2  headers                   List of non-empty values from row 1
  F3  sample_rows               Rows 2–6 as raw values
  F4  formula_samples           Up to 30 formulas beginning with "="
  F5  formula_sheet_refs        Sheet names extracted from formulas
  F6  all_values_flat           Up to 300 unique body strings
  F7  is_active_sheet           Boolean — workbook active sheet?
  F8  candidate_code_columns    Suspected section-code/account-code columns
  F9  candidate_desc_columns    Suspected account-description columns
  F10 candidate_final_columns   Suspected FINAL amount columns
  F11 repetition_stats          Repetition / uniqueness stats per candidate column
  F12 adjacency_stats           Distance between code and description candidates
  F13 numeric_column_stats      Numeric density and null density per amount column
  F14 incoming_formula_targets  Which local columns are referenced by upstream formulas
  F15 outgoing_formula_targets  Which external sheets/columns this sheet points to
  F16 workbook_sheet_names      Exact list of worksheet tab names in the workbook

Extraction rules:
  • Open workbook with data_only=False so formulas are visible.
  • Formula strings are primary evidence.
  • Do not trust sheet names alone.
  • Do not trust highlighted titles alone.
  • Candidate FINAL columns may be discovered by:
      1. explicit header text,
      2. being the target of upstream formulas,
      3. being the dominant numeric ending-balance column.
  • If there is only one strong numeric ending-balance column, it is the FINAL
    candidate even if the header is weak.
  • Code and description may be separate columns OR combined in one column.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  LAYER 1 — Binary activations
# ──────────────────────────────────────────────────────────────────────────────

NN_LAYER_1 = """
╔══════════════════════════════════════════════════════════╗
║  LAYER 1 — Binary Signal Activations                    ║
╚══════════════════════════════════════════════════════════╝

Compute each signal as exactly 0 or 1. No partial activations.

COA_SIGNAL
  ON if all_values_flat contains at least 3 of these section strings
  (case-insensitive):
    "assets"
    "current assets"
    "long-term assets"
    "liabilities and equity"
    "current liabilities"
    "long-term liabilities"
    "equity"

FORMULA_SIGNAL
  ON if formula_samples is non-empty.

CROSS_REF_SIGNAL
  ON if this sheet references at least one other sheet.

REFERENCED_BY_SIGNAL
  ON if at least one other sheet references THIS sheet.

COMPANY_COLUMN_SIGNAL
  ON if there is evidence of at least one company/currency-style amount column.
  Positive evidence may include:
    • headers / labels containing "NIS", "$", "DOLLAR", "USD", "ILS", "INC", "LTD"
    • company abbreviations inferred from layout
    • repeated parallel amount columns aligned under company-like headings
  IMPORTANT:
    • lack of explicit currency text does NOT prove absence of a company column
    • this signal is business-structural, not strict string matching only

AJE_SIGNAL
  ON if headers, labels, or structure indicate:
    "AJE", "ADJUSTING", "ADJUSTMENT"

CONSOLIDATE_SIGNAL
  ON if headers, labels, or structure indicate:
    "CONSOLIDATE", "CONSOL", "TOTAL"

HAS_CODE_COLUMN
  ON if there exists a strong section-code/account-code candidate such that:
    • values are mostly alphanumeric / code-like
    • values do not behave like free-text descriptions
    • repetitions are limited
  Selection rule:
    If several code candidates exist, prefer the one with the fewest repetitions
    and the one closest to the description column.

HAS_DESCRIPTION_COLUMN
  ON if there exists a strong account-description candidate such that:
    • values behave like account descriptions
    • repetitions are limited
    • the column is adjacent or close to a code candidate
  Selection rule:
    If several description candidates exist, prefer the one with fewer repetitions
    and stronger adjacency to the chosen code column.

HAS_FINAL_COLUMN
  ON if there exists at least one strong FINAL amount column candidate.
  A FINAL candidate may be identified by ANY of:
    • header contains "FINAL"
    • header contains "TRIAL BALANCE" or "TB"
    • upstream formulas clearly pull values from this column
    • this is the dominant ending-balance numeric column even if header is generic

FINAL_REFERENCE_SIGNAL
  ON if upstream reporting/intermediate formulas reference this sheet in a way
  consistent with pulling balances from a specific amount column.

TB_REFERENCE_SIGNAL
  ON if this sheet behaves like a source sheet that feeds COA balances upward.

STAGING_ROLE_SIGNAL
  ON if sheet name, labels, or structure indicate staging / adjustment behavior:
    "aje", "adjusting", "adjustments", "elimination",
    "mapping", "bridge", "rollforward", "support", "schedule"

VISIBLE_STATEMENT_TAB_SIGNAL
  ON if the real workbook tab name itself strongly suggests a final visible
  statement-family header sheet, such as:
    `FS`, `BS`, `P&L`, `PL`, `CF`
  IMPORTANT:
    • this signal is only supportive
    • it never overrides hard gates by itself
    • it is especially important when the business truth is split across
      multiple real header tabs

HIDDEN_SIGNAL
  ON if is_hidden = true

Layer 1 rule:
  If uncertain, default to 0.
  Names are hints; formulas and structure are authority.
  However, any referenced or candidate sheet name must still be an exact real
  workbook tab name from Layer 0.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  LAYER 2 — Pattern logic
# ──────────────────────────────────────────────────────────────────────────────

NN_LAYER_2 = """
╔══════════════════════════════════════════════════════════╗
║  LAYER 2 — Pattern Logic                                ║
╚══════════════════════════════════════════════════════════╝

Combine Layer-1 signals into composite patterns.

FS_PATTERN  (Full Financial Statement — main reporting candidate)
  = COA_SIGNAL
    AND FORMULA_SIGNAL
    AND CROSS_REF_SIGNAL
    AND COMPANY_COLUMN_SIGNAL
    AND NOT HIDDEN_SIGNAL
    AND NOT HAS_CODE_COLUMN

Interpretation:
  The sheet is a reporting/output sheet aggregating values from lower-level sheets.

PARTIAL_FS_PATTERN  (Partial FS evidence)
  = COA_SIGNAL
    AND (FORMULA_SIGNAL OR CROSS_REF_SIGNAL)
    AND NOT HIDDEN_SIGNAL

Interpretation:
  Reporting structure exists but full confirmation requires graph support.

TB_PATTERN  (TB / card / source sheet)
  = HAS_CODE_COLUMN
    AND HAS_DESCRIPTION_COLUMN
    AND HAS_FINAL_COLUMN
    AND NOT HIDDEN_SIGNAL

Interpretation:
  This is a source/card sheet containing the detailed cards/accounts that feed
  the reporting structure.

Important:
  • COMPANY_COLUMN_SIGNAL may be ON or OFF
  • COA_SIGNAL may be ON or OFF
  • TB is NOT disqualified by having company-like headers
  • the old rule "NOT COMPANY_COLUMN_SIGNAL" is forbidden

STRONG_TB_PATTERN
  = TB_PATTERN
    AND (FINAL_REFERENCE_SIGNAL OR TB_REFERENCE_SIGNAL OR REFERENCED_BY_SIGNAL)

Interpretation:
  Preferred TB candidate because both structure and formula-graph behavior agree.

STAGING_PATTERN
  = STAGING_ROLE_SIGNAL
    AND (AJE_SIGNAL OR FORMULA_SIGNAL OR CROSS_REF_SIGNAL)

Interpretation:
  Sheet is a staging / adjustment / bridge layer, not the final main report.

HEADER_SHEET_PATTERN
  = NOT HIDDEN_SIGNAL
    AND NOT STAGING_PATTERN
    AND (
      FS_PATTERN
      OR (
        VISIBLE_STATEMENT_TAB_SIGNAL
        AND PARTIAL_FS_PATTERN
      )
    )

Interpretation:
  The sheet is a real visible final statement-family header tab.
  This pattern exists to preserve business truth in workbooks where the final
  reporting output is split across real tabs such as `BS` and `P&L`.

Priority rules:
  • FS_PATTERN overrides PARTIAL_FS_PATTERN for main-sheet confidence.
  • TB_PATTERN blocks a sheet from being the main reporting sheet.
  • STAGING_PATTERN blocks a sheet from being the final main reporting sheet
    when graph evidence shows it serves as an upstream adjustment source.
  • HEADER_SHEET_PATTERN preserves real visible statement tabs as valid final
    header-sheet answers even when the workbook has more than one such tab.
  • If a sheet shows both FS-like and TB-like evidence, source structure wins
    only when code/description/final structure clearly dominates AND graph places
    the sheet upstream from reporting output.
  • Do not demote a visible statement-family tab to pure TB/source status
    merely because it contains formulas or balance-pull structure if its business
    role is a final visible header sheet.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  LAYER 3 — Context graph
# ──────────────────────────────────────────────────────────────────────────────

NN_LAYER_3 = """
╔══════════════════════════════════════════════════════════╗
║  LAYER 3 — Context Graph                                ║
╚══════════════════════════════════════════════════════════╝

Build a directed dependency graph across ALL sheets.

Graph nodes may only be real workbook sheet names discovered in Layer 0.
If a title, caption, or business label appears in reasoning but is not an exact
workbook tab name, it must not become a graph node.

Graph construction:
  For every sheet X, create edges X → Y for each Y referenced by formulas in X,
  but only if Y is an exact workbook tab name from Layer 0.
  After building, collect incoming_refs and outgoing_refs for each sheet.
  Do not create synthetic graph nodes from semantic interpretation.

Attention rules:

[A1 — Directional identity]
  A sheet with outgoing references is a reporting candidate.
  A sheet with incoming references is a source candidate.
  Valid business direction:
    main / presentation → intermediate / staging → TB
  Invalid dominance for final-report identity:
    TB → FS
    staging → FS as final winner

[A2 — Reference concentration]
  The sheet most frequently referenced by others is a strong source/TB candidate.
  The sheet with the strongest outward referencing is a strong FS candidate.

[A3 — Intermediate chain detection]
  If X → Y → Z:
    X = main reporting candidate
    Y = intermediate / staging candidate
    Z = TB/source candidate
  Record the chain explicitly.

[A4 — Single-company-column CONSOLIDATE logic]
  If COMPANY_COLUMN_SIGNAL = 1 but only one company-like amount column exists:
    inspect referenced intermediate sheets.
    If an intermediate sheet contains COA structure and multiple company columns,
    set consolidate = true for the current sheet.

[A5 — Main-to-TB path validation]
  Every element of path_to_tb must be an exact workbook tab name.
  If any element is not a real sheet tab, path_valid = false.
  For every strong FS candidate, find the strongest valid path:
    • direct: FS → TB
    • indirect: FS → INTERMEDIATE → TB
    • indirect: FS → STAGING → TB
  Record path_to_tb and path_valid.

[A6 — Functional FINAL detection]
  If upstream formulas from a main/intermediate sheet repeatedly target one
  amount column in a source sheet, treat that target column as FINAL evidence
  even if header text is weak.

[A7 — AJE source-role detection]
  If a sheet primarily feeds AJE / adjustment columns of another sheet,
  mark aje_source_role = true.
  Such a sheet is staging/support, not the final reporting output.

[A8 — Active sheet boost]
  If is_active_sheet = true AND FS_PATTERN = 1, set attention_boost = true.
  This only mildly strengthens technical confidence and never bypasses gates.

[A9 — Multi-header preservation]
  If multiple real visible statement-family tabs each behave like final output
  slices of the workbook, preserve them as separate final header candidates.
  Do not collapse them into an inferred umbrella node.

Layer 3 output per sheet:
  {
    "outgoing_refs": [...],
    "incoming_refs": [...],
    "role_in_graph": "FS" | "TB" | "INTERMEDIATE" | "STAGING" | "UNKNOWN",
    "consolidate": true/false,
    "attention_boost": true/false,
    "aje_source_role": true/false,
    "path_to_tb": ["<sheet>", ...] | [],
    "path_valid": true/false
  }
"""

# ──────────────────────────────────────────────────────────────────────────────
#  LAYER 4 — Hard gates
# ──────────────────────────────────────────────────────────────────────────────

NN_LAYER_4 = """
╔══════════════════════════════════════════════════════════╗
║  LAYER 4 — Hard Gates (Main-Sheet Firewalls)            ║
╚══════════════════════════════════════════════════════════╝

Hard gates apply to MAIN reporting-sheet candidacy only.
They do NOT, by themselves, block TB selection unless explicitly stated.

GATE_1  [HIDDEN FIREWALL]
  Condition: HIDDEN_SIGNAL = 1
  Action: BLOCK from main-sheet candidacy.
  Hidden sheets can never be the main reporting sheet.

GATE_2  [NO COMPANY / CONSOLIDATE FIREWALL]
  Condition: COMPANY_COLUMN_SIGNAL = 0
             AND CONSOLIDATE_SIGNAL = 0
             AND consolidate = false
  Action: BLOCK from technical FS candidacy.
  Classification: TECHNICAL
  Explanation:
    This protects against generic summaries and false-positive report tabs.
    Layer 6 may reconsider this only for strong presentation sheets.

GATE_3  [TB FIREWALL]
  Condition: TB_PATTERN = 1
  Action: BLOCK from main-sheet candidacy.
  Classification: CRITICAL
  Explanation:
    A TB/card sheet is a source, not the final reporting output.

GATE_4  [PURE SOURCE FIREWALL]
  Condition: role_in_graph = "TB"
             AND FS_PATTERN = 0
             AND PARTIAL_FS_PATTERN = 0
  Action: BLOCK from main-sheet candidacy.
  Classification: CRITICAL

GATE_5  [STAGING FIREWALL]
  Condition: STAGING_PATTERN = 1
             AND (role_in_graph = "STAGING" OR aje_source_role = true)
  Action: BLOCK from main-sheet candidacy.
  Classification: CRITICAL
  Explanation:
    AJE / mapping / bridge / rollforward sheets may be structurally rich and may
    point toward TB, but they are staging layers, not the user-facing FS output.

Gate evaluation order:
  GATE_1 → GATE_2 → GATE_3 → GATE_4 → GATE_5

Important interpretation safeguard:
  A visible statement-family header tab must not be forced into GATE_3 purely
  because it contains code/description/final pull structure unless the graph and
  business role clearly show that it is truly a terminal TB/source rather than
  a final visible statement sheet.

Layer 4 output per sheet:
  {
    "passed": true/false,
    "blocked_by": "GATE_1" | "GATE_2" | "GATE_3" | "GATE_4" | "GATE_5" | null
  }
"""

# ──────────────────────────────────────────────────────────────────────────────
#  LAYER 5 — Technical main-sheet confidence
# ──────────────────────────────────────────────────────────────────────────────

NN_LAYER_5 = """
╔══════════════════════════════════════════════════════════╗
║  LAYER 5 — Technical Main-Sheet Confidence              ║
╚══════════════════════════════════════════════════════════╝

Only sheets that passed Layer 4 participate in main-sheet selection.

Pre-softmax logit per passing sheet:
  z(sheet) = (
    FS_PATTERN            × 2.40 +
    PARTIAL_FS_PATTERN    × 1.10 +
    CROSS_REF_SIGNAL      × 0.45 +
    COMPANY_COLUMN_SIGNAL × 0.40 +
    CONSOLIDATE_SIGNAL    × 0.35 +
    attention_boost       × 0.20
  )

Negative main-sheet effects:
  • AJE_SIGNAL alone does NOT make a sheet more likely to be the final main sheet.
  • If staging evidence exists but the sheet still passed, staging-like evidence
    should reduce interpretation confidence during review, not promote the sheet.

True softmax:
  confidence(sheet_i) = exp(z_i / τ) / Σ_j exp(z_j / τ)
  Recommended τ = 0.75

Why this matters:
  • true softmax sharply separates strong FS candidates from noisy partial sheets
  • a fully qualified FS candidate should dominate weaker partial candidates
  • simple ratio normalisation is forbidden

Decision thresholds:
  confidence ≥ 0.70  → CONFIRMED main sheet
  confidence ≥ 0.40  → POSSIBLE main sheet
  confidence < 0.40  → weak / uncertain

Tie-breaking:
  1. prefer the sheet with stronger COA section coverage
  2. prefer the sheet with stronger valid path to TB
  3. prefer the sheet that functions as the reporting origin rather than staging
  4. prefer a real visible statement-family header tab over a non-header support tab
  5. if still tied, manual review required

Layer 5 output:
  TECHNICAL main-sheet winner only.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  LAYER 6 — Business arbitration
# ──────────────────────────────────────────────────────────────────────────────

NN_LAYER_6 = """
╔══════════════════════════════════════════════════════════╗
║  LAYER 6 — Business Arbitration (Main Sheet)            ║
╚══════════════════════════════════════════════════════════╝

Purpose:
  Resolve the gap between:
    • technical main sheet
    • human-facing final reporting sheet
    • the real header-sheet set when final reporting is split across multiple tabs

Key principles:
  • do not trust sheet name alone
  • do not trust highlighted title alone
  • semantic logic operates only among safe candidates
  • never override into hidden sheets, TB sheets, or staging sheets
  • Layer 6 may arbitrate only among real workbook tab names
  • business arbitration may not invent a new sheet name

BUSINESS SIGNALS

CANONICAL_FS_TITLE_SIGNAL
  ON if sheet_name or dominant title contains strong FS wording:
    "balance sheet", "balance sheets",
    "statement of operations", "statement of income",
    "profit and loss", "p&l",
    "cash flow", "cash flows", "statement of cash flows",
    "change in equity", "stockholders' equity",
    "financial statements", "report"

PRESENTATION_LAYOUT_SIGNAL
  ON if the sheet looks like a human-facing FS layout:
    • readable line items
    • statement sections
    • period / year columns
    • low raw-code dominance

FINAL_OUTPUT_ROLE_SIGNAL
  ON if the sheet is likely the user-facing final report.

HEADER_SET_ROLE_SIGNAL
  ON if the workbook’s final human-facing reporting output is clearly spread
  across multiple real visible statement-family tabs, such as `BS` and `P&L`.

SHEET TYPE CLASSIFIER

  SOURCE_TB
    If TB_PATTERN = 1
    OR graph/source evidence dominates

  ADJUSTMENT_STAGING
    If STAGING_PATTERN = 1
    OR role_in_graph = "STAGING"
    OR aje_source_role = true

  REPORTING_FS
    If FS_PATTERN = 1 and not explicitly staging
    OR strong presentation/report evidence exists with COA support

  HEADER_FS
    If the sheet is a real visible final statement-family tab that belongs to
    the final header-sheet set, even when the workbook has multiple such tabs

  INTERMEDIATE_CONSOLIDATION
    If the sheet bridges reporting to source but is not itself the final report

  AUXILIARY_SCHEDULE
    If it has FS-like content but behaves like a schedule/support tab

  UNKNOWN
    Otherwise

DISQUALIFICATION CLASS

  CRITICAL
    If blocked_by ∈ {GATE_1, GATE_3, GATE_4, GATE_5}

  TECHNICAL
    If blocked_by = GATE_2 only and reporting evidence is strong

  NONE
    If passed = true

BUSINESS ARBITRATION RULES

BA1 — Technical winner
  Highest-confidence passing sheet from Layer 5.

BA2 — Presentation candidate
  Strongest safe REPORTING_FS candidate visible to the user,
  chosen only from real workbook tab names.

BA3 — Safe override
  Override only if ALL are true:
    • technical winner is not the clearest human-facing final report
    • presentation candidate is REPORTING_FS or HEADER_FS
    • presentation candidate is not critically blocked
    • presentation candidate shows FINAL_OUTPUT_ROLE_SIGNAL = 1

BA4 — No unsafe override
  Never override to:
    • SOURCE_TB
    • ADJUSTMENT_STAGING
    • hidden sheets
    • pure incoming-only sources
    • nonexistent sheet names
    • semantic umbrella labels
    • title-derived labels that are not actual tabs

BA5 — Dual truth preservation
  Preserve both:
    • technical_main_sheet
    • presentation_main_sheet
  final_main_sheet may equal either one depending on override.

BA6 — Sheet-name fidelity
  technical_main_sheet, presentation_main_sheet, business_main_sheet,
  and final_main_sheet must all be exact workbook tab names.
  If a candidate is not an exact workbook tab name, it is invalid and must
  be discarded.

BA7 — Header-sheet set preservation
  If the final human-facing output is clearly split across multiple real visible
  statement-family tabs, preserve them as:
    "header_sheets": ["BS", "P&L", ...]
  Do not collapse them into a synthetic parent.
  Do not discard one valid header tab merely because another valid header tab
  also exists.

Layer 6 output:
  {
    "technical_main_sheet": "<sheet>" | null,
    "presentation_main_sheet": "<sheet>" | null,
    "business_main_sheet": "<sheet>" | null,
    "final_main_sheet": "<sheet>" | null,
    "header_sheets": ["<sheet>", ...],
    "decision_mode": "technical_default" | "business_override" | "no_valid_sheet"
  }
"""

# ──────────────────────────────────────────────────────────────────────────────
#  LAYER 7 — TB / card-sheet validation
# ──────────────────────────────────────────────────────────────────────────────

NN_LAYER_7 = """
╔══════════════════════════════════════════════════════════╗
║  LAYER 7 — TB / Card-Sheet Validation                   ║
╚══════════════════════════════════════════════════════════╝

Purpose:
  Identify the TB/card sheet that structurally feeds the main reporting sheet.
  The TB/card sheet and every main-to-TB path node must be exact workbook tab names.
  Synthetic bridge labels or inferred semantic nodes are forbidden.

TB business definition:
  A TB sheet contains the card/account details whose summed values — including
  arithmetic operations and optional AJE adjustments — feed the COA balances
  shown in the reporting sheet.

A valid TB candidate should strongly satisfy:
  • HAS_CODE_COLUMN = 1
  • HAS_DESCRIPTION_COLUMN = 1
  • HAS_FINAL_COLUMN = 1

Strong reinforcing evidence:
  • FINAL_REFERENCE_SIGNAL = 1
  • TB_REFERENCE_SIGNAL = 1
  • REFERENCED_BY_SIGNAL = 1
  • appears at the end of a valid main → intermediate/staging → TB chain
  • numeric amount behaviour is consistent with balance pull-up

TB selection rules:

TB1 — Structural priority
  Prefer STRONG_TB_PATTERN over plain TB_PATTERN.

TB2 — Graph priority
  Every candidate path must contain only real workbook tab names.
  If a path includes a nonexistent or invented node, that path is invalid.
  Prefer a TB candidate reachable from the final main sheet by:
    • direct path: main → TB
    • indirect path: main → intermediate → TB
    • indirect path: main → staging → TB

TB3 — Code-column quality
  If several code-like columns exist, prefer the one with:
    • fewest repetitions
    • highest unique-to-total ratio
    • strongest adjacency to description column
    • strongest code-like formatting (alphanumeric, no spaces)

TB4 — Description-column quality
  If several description candidates exist, prefer the one with:
    • fewer repetitions
    • proximity to code column
    • account-description behaviour rather than generic labels

TB5 — FINAL-column quality
  Prefer the amount column that is:
    • referenced by upstream formulas
    • or acts as the principal ending-balance numeric column
  Header text helps but is not required.
  If there is no formula but the column is the dominant ending-balance numeric
  column, treat it as FINAL.

TB6 — AJE interpretation
  AJE may exist as:
    • one AJE column
    • two AJE columns: CREDIT and DEBIT
  If no AJE exists, treat AJE = 0.
  If only FINAL exists, FINAL remains the effective total amount source.

TB7 — Hidden restriction
  Hidden sheets should not be selected as the TB/card output unless no visible
  evidence-backed TB exists and manual review is required.

Layer 7 output:
  {
    "is_card_sheet": "<tb_sheet_name>" | null,
    "technical_tb_sheet": "<tb_sheet_name>" | null,
    "relationship": {
      "main_to_tb_path": ["<main>", "...", "<tb>"],
      "path_valid": true/false
    }
  }
"""

# ──────────────────────────────────────────────────────────────────────────────
#  DECISION PROTOCOL
# ──────────────────────────────────────────────────────────────────────────────

NN_DECISION_PROTOCOL = """
╔══════════════════════════════════════════════════════════╗
║  DECISION PROTOCOL — Full 7-Layer Execution             ║
╚══════════════════════════════════════════════════════════╝

Execute these steps in strict order for every workbook:

  Step 0 — F0:
    Enforce exact workbook-tab identity.
    No synthetic sheet names are allowed anywhere.

  Step 1 — L0:
    Extract raw workbook features for every sheet.
    Build the canonical workbook sheet-name universe.
    Only exact names from this set may appear in any later output.

  Step 2 — L1:
    Compute all binary structural signals.

  Step 3 — L2:
    Compute FS_PATTERN, PARTIAL_FS_PATTERN, TB_PATTERN,
    STRONG_TB_PATTERN, STAGING_PATTERN, and HEADER_SHEET_PATTERN.

  Step 4 — L3:
    Build dependency graph.
    Assign graph roles.
    Detect main → intermediate/staging → TB chains.
    Record path_to_tb and path_valid.
    Detect functional FINAL-column evidence.

  Step 4b — Sheet-name validation:
    Remove any candidate, evidence key, graph node, or path node that is not
    an exact workbook tab name.

  Step 5 — L4:
    Apply hard gates for MAIN-sheet candidacy only.

  Step 6 — L5:
    Compute technical main-sheet confidence using TRUE SOFTMAX.
    Identify technical main-sheet winner.

  Step 7 — L6:
    Perform business arbitration for the final main reporting sheet.
    Preserve technical_main_sheet and presentation_main_sheet.
    Also preserve header_sheets when the reporting truth is split across
    multiple real statement-family tabs.

  Step 8 — L7:
    Select the strongest TB/card sheet using structural and graph validation.
    Do not rely on sheet name alone.
    Do not rely on highlighted titles alone.

  Step 9 — Final output:
    Emit final main sheet, header sheets, TB sheet, and validated relationship
    using only exact workbook tab names.
    If no exact workbook tab name supports a proposed candidate, emit null
    instead of inventing a name.

Final output contract:
  {
    "main_sheet_exists": true/false,
    "main_sheet_name": "<exact workbook tab name>" | null,
    "header_sheets": ["<exact workbook tab>", ...],
    "is_card_sheet": "<exact workbook tb tab name>" | null,
    "technical_main_sheet": "<exact workbook tab name>" | null,
    "presentation_main_sheet": "<exact workbook tab name>" | null,
    "technical_tb_sheet": "<exact workbook tab name>" | null,
    "decision_mode": "technical_default" | "business_override" | "business_override_with_tb_validation" | "no_valid_sheet",
    "relationship": {
      "main_to_tb_path": ["<exact workbook tab>", "...", "<exact workbook tb tab>"],
      "path_valid": true/false
    }
  }

Detector rule:
  The detector result is a hint only.
  It must pass structural validation independently.
  Never return detector output without NN validation.

Sheet-name rule:
  Even if a detector, title, or business interpretation suggests a compelling
  reporting label, never return it unless it exactly matches a real workbook
  tab name discovered in Layer 0.

Header-sheet rule:
  If the real final reporting output is split across multiple real visible
  header tabs — for example `BS` and `P&L` — return those tabs.
  Do not replace them with an umbrella concept.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  FULL PROMAT BLOCK
# ──────────────────────────────────────────────────────────────────────────────

FULL_NN_PROMAT = f"""
{'═' * 62}
  NEURAL PROMAT SYSTEM  v6.1
  Deterministic multi-layer architecture for:
    • main reporting sheet detection
    • real tab-name fidelity
    • header-sheet-set preservation
    • business/presentation arbitration
    • TB/card-sheet validation
  Structure first. Real tab names are identity. Titles are hints only.
{'═' * 62}

{NN_SHEET_NAME_FIDELITY}

{NN_LAYER_0}

{NN_LAYER_1}

{NN_LAYER_2}

{NN_LAYER_3}

{NN_LAYER_4}

{NN_LAYER_5}

{NN_LAYER_6}

{NN_LAYER_7}

{NN_DECISION_PROTOCOL}
"""