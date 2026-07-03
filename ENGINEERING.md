# StockIQ Pro Engineering Constitution

This document defines the core principles, standards, and rules governing all development in the StockIQ Pro project. It acts as a guide and a rulebook for both human engineers and AI coding agents.

---

## 🎯 Mission

StockIQ Pro aims to demonstrate professional software engineering applied to quantitative finance.

The objective is not to build the largest stock analysis platform. The objective is to build a platform that is:
* **Technically accurate**: Financial math, indicators, and predictions must be statistically and mathematically sound.
* **Architecturally clean**: Decoupled, modular boundaries with single-responsibility components.
* **Maintainable**: Straightforward logic that any future engineer can immediately understand and extend.
* **Performant**: Smart caching, optimized computational structures, and minimal latency overhead.
* **Well-documented**: Accurate system diagrams, clean API docs, and transparent codebase comments.
* **Production-quality**: Production-ready code quality, robust error resilience, and automated testing.

Every engineering decision should move the project closer to these goals.

---

## 🚦 Core Philosophy & Stability

### Stability First
Do not rewrite code simply because a different implementation is possible. If an implementation is:
* Correct
* Maintainable
* Performant
* Readable

leave it unchanged. Prefer incremental improvements over large rewrites. Preserve git history where practical.

### Architecture Before Features
When deciding between:
* Adding a new feature
* Improving the architecture

prefer improving the architecture first. A clean architecture enables future features. A messy architecture slows every future feature.

### Evidence-Based Engineering
Do not optimize based on assumptions. Support engineering decisions with evidence whenever possible.
* Benchmark before optimizing.
* Profile before rewriting.
* Measure latency before caching.
* Identify duplication before refactoring.

Engineering decisions should be driven by measurable improvements.

---

## 🤖 AI Agent Operating Rules

The AI is expected to behave like a senior software engineer. Before making changes, it should:
* Inspect the existing implementation.
* Understand the current architecture.
* Identify dependencies and potential risks.
* Explain trade-offs clearly.

Never assume. Always inspect. Never replace working implementations without clear engineering justification. When uncertain, investigate the codebase rather than guessing.

---

## 📐 Engineering Principles

### Single Responsibility & Modular Design
Each module should have one responsibility. Avoid "God files" or massive utility files containing unrelated logic.
Prefer a decoupled folder structure over monolithic files:
* `routers/`: API route boundary, parameter validation, and endpoint definitions.
* `services/`: Business logic, mathematical formulas, and ML pipelines.
* `utils/`: Reusable, narrowly focused helpers (caching, limiters, constants).
* `tests/`: Automated unit and integration tests.

### Avoid Duplication (DRY)
Extract reusable logic rather than duplicating calculations, data validation, or API requests. Keep fundamental data extraction and calculations unified.

### Caching and Performance
Use the thread-safe in-memory Time-To-Live (TTL) cache decorator (`@cache_ttl`) for expensive database or network roundtrips. Do not introduce computational complexity without measurable latency or resource benefits.

---

## 🔒 Code Quality & Defensive Coding

* **Graceful Degradation & Fallbacks**: Ensure that network calls to third-party APIs (like Yahoo Finance) are wrapped in try-except blocks with clean default states.
* **No Placeholders**: Never deploy placeholder variables, dummy results, or stubbed endpoints in production. Use real data paths.
* **No Over-Engineering**: Avoid excessive abstractions and premature optimizations. Choose clarity over cleverness.

---

## ⚖️ Engineering Priorities

When multiple solutions exist, prioritize in this order:

1. **Correctness**
2. **Reliability**
3. **Maintainability**
4. **Simplicity**
5. **Performance**
6. **Extensibility**
7. **New Features**

Features are valuable only when built on a reliable foundation.

---

## 🏁 Definition of Done & Quality Gates

### Quality Gates
A task is complete only if:
* [ ] Code compiles without errors or warnings.
* [ ] Existing functionality is fully preserved.
* [ ] No duplicate logic is introduced.
* [ ] No circular imports are created.
* [ ] No unnecessary dependencies are added.
* [ ] Unit tests pass successfully (`pytest` in `backend`).
* [ ] Code adheres strictly to project architecture.
* [ ] Public APIs remain compatible unless explicitly changed.
* [ ] Documentation is updated if system behavior changes.

### Definition of Done
A task is considered complete only when:
* Functionality works correctly.
* Architecture remains clean.
* Tests succeed.
* Documentation reflects the change.
* Unnecessary complexity has not been introduced.

Completion is measured by software quality, not by lines of code written.

---

## 🚀 Communication & Reporting Style

When interacting or working on a task:
1. **Audit first** and report findings.
2. **Explain the reasoning** behind any proposed change.
3. **Present the implementation plan** and wait for feedback if necessary.
4. **Implement only what is required**.
5. **Verify correctness** and run unit tests.
6. **Summarize improvements** objectively.
