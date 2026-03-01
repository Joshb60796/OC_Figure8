---
name: code-review
description: Comprehensive code review of changes, files, PR diffs, or code snippets. Always output in clean Markdown with summary tables, severity ratings, and prioritized actionable fixes.
category: code-quality
version: 1.2
compatibility: opencode
---

# Code Review Skill

## When to activate
Use this skill whenever the user says:
- "review", "code review", "review this", "review the changes"
- "review the PR", "review these files", "what do you think of this code"
- "perform a full review", "security review", "quality review"

## Review Process (strict order)
1. **Understand context** — Look at git diff if available, or read the full files.
2. **Run full checklist** — Evaluate every category below.
3. **Output format** — Always use this exact structure (no exceptions).

## Output Format (exactly like this)

### Summary
| Category          | Issues Found | Severity | Score (1-10) |
|-------------------|--------------|----------|--------------|
| Security          | ...          | ...      | ...          |
| Bugs & Logic      | ...          | ...      | ...          |
| Performance       | ...          | ...      | ...          |
| Maintainability   | ...          | ...      | ...          |
| Style & Readability | ...        | ...      | ...          |
| Testing           | ...          | ...      | ...          |
| **Overall**       | **...**      | **...**  | **...**      |

### Critical Issues (fix before merge)
- [ ] Issue 1...
- [ ] Issue 2...

### High Priority Recommendations
- [ ] ...

### Medium / Low Suggestions

### Positive Highlights
(Always end with at least 2-3 things done well)

## Full Checklist (check every item)
**Security**
- SQL/NoSQL injection, XSS, CSRF
- Hardcoded credentials/secrets
- Authentication/authorization bypass
- Insecure dependencies or outdated packages
- File upload / path traversal risks
- Logging of sensitive data

**Bugs & Correctness**
- Off-by-one errors, null/undefined handling
- Race conditions, async/await mistakes
- Edge cases and input validation
- Error handling & recovery

**Performance**
- N+1 queries, unnecessary loops
- Memory leaks, heavy operations in hot paths
- Inefficient algorithms or data structures

**Maintainability & Best Practices**
- Single Responsibility, DRY, KISS
- Naming, comments, documentation
- Consistent style with project conventions (from AGENTS.md)
- Proper modularity and separation of concerns

**Testing**
- Are tests present and covering changes?
- Are tests meaningful or just smoke tests?

**Other**
- Accessibility (if UI)
- Internationalization
- Licensing/copyright

## Final Instructions
- Be constructive and specific (quote exact code lines when possible)
- Suggest exact fixes or refactors
- Never make changes yourself unless user says "apply" or "fix"
- End with "Ready to apply any fixes?" or "Shall I create a commit with improvements?"

Always load this skill automatically on review requests.
