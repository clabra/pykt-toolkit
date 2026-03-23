# MATH_TODO: MDPI Math and Technical Compliance

This document lists the identified non-compliance points and formatting errors in the manuscript `applsci-4169306-done.tex` that need to be addressed to meet MDPI standards and maintain consistency with author-stated rules.

## Priority Checklist

### 1. Bayesian Parameter Subscripts (Upright Labels Rule) [Done]
- **Problem**: Subscripts $L_0, T, G, S$ are consistently in italics in the text and captions, which violates the "labels are upright" rule (MDPI standard and author's own claim on line 168).
- **Locations**:
    - Lines 389, 391 (Text)
    - Lines 704, 706 (Text)
    - Line 714 (Figure 4 caption)
    - Line 719 (Figure 5 caption)
    - Line 812 (Text)
- **Recommendation**: Standardize all mentions to:
    - `$P(L_0)$` → `$P(\text{L}_0)$`
    - `$P(T)$` → `$P(\text{T})$`
    - `$P(G)$` → `$P(\text{G})$`
    - `$P(S)$` → `$P(\text{S})$`

### 2. Incorrect Total Equation Count [Done]
- **Problem**: Line 337 states the manuscript contains 14 numbered equations (Equations (1)--(14)).
- **Status**: The document actually contains **16** numbered equations (Eq 15 is on line 524, Eq 16 is on line 530).
- **Recommendation**: Update line 337 to correctly state **16 equations (Equations (1)--(16))**.

### 3. Missing Equation Punctuation [Done]
- **Problem**: Displayed equations are part of sentences and must have terminal punctuation.
- **Missing Periods/Commas**:
    - **Equation (9)** (line 473): Missing period at end.
    - **Equation (10)** (line 478): Missing period at end.
    - **Equation (11)** (line 483): Missing period at end.
    - **Equation (15)** (line 525): Missing comma (followed by "where" on line 527).

### 4. Numbers as Words [Done]
- **Problem**: Single-digit numbers (one to nine) should be spelled out when they are not measurements, indices, or variables.
- **Location**: Line 544 (*"5 NVIDIA Tesla GPUs"*).
- **Decision**: We interpret these as **quantitative values directly tied to technical entities** ("CPU cores", "GPUs"), which allows for the use of digits for consistency. No change was needed as the manuscript already used digits (24 and 5).
- **Recommendation**: (Already handled).

### 5. Bold Table Footer Formatting [Done]
- **Problem**: The "Mean" row label in Table 4 is bolded, which violates MDPI's preference for plain-text footer labels.
- **Location**: Line 674 (`\textbf{\hl{Mean}}`).
- **Recommendation**: Remove the `\textbf` command. [Handled by USER].

### 6. Bibliography: Missing Access Date [Done]
- **Problem**: The KDD Cup 2010 reference has an empty placeholder for the access date.
- **Location**: Line 1231 (`\hl{accessed on}`).
- **Recommendation**: Provide the actual access date. [Handled by USER: 28 December 2025].

---

## Final Verification Steps (Post-Fix)
- [ ] Check if `RQ1`, `RQ2`, `RQ3` identifiers are consistently formatted (plain text vs italic vs math mode).
- [ ] Verify that no new equations were added that would invalidate the count again.
- [ ] Ensure all `\hl{}` and `\highlighting{}` tags are preserved or removed only when instructed.
