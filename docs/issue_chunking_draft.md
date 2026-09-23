# Draft: relevance-aware chunking request for google/langextract

Unpublished draft for an upstream feature request, kept here so the numbers and
framing survive outside a chat session. It follows up on
[google/langextract#358](https://github.com/google/langextract/issues/358),
which was closed 2026-09-20 as too broad, with the maintainer asking for focused
issues naming a concrete example and provider. Relevance-aware chunking is the
only item from #358 never addressed upstream.

To file it: <https://github.com/google/langextract/issues/new>, **Feature
Request** template — the body below already matches its required sections, and
the template applies the `enhancement` and `needs triage` labels automatically.

Figures verified 2026-09-23 against `exemplo/62941946000100-REG12082026V01-001284583.pdf`
at `max_char_buffer=3000`: 159,629 → 50,043 characters (68.7% reduction).

---

**Title** (the template asks for the `Request:` prefix)

`Request: allow callers to filter or score chunks before inference`

---

## Describe the overall idea and motivation

Long, structured documents are chunked uniformly, so cost scales with document length rather than with where the answers actually are. I would like a way to skip chunks that cannot contain a target entity, without reimplementing chunking.

The concrete case, and the numbers below, come from Brazilian investment fund regulations filed with CVM (the securities regulator) under Resolution 175/2022 — standardized ~100-page PDFs. The working document is a FIDC regulation, a public filing.

- Provider: `gemini-2.5-flash`, Gemini API free tier, 15 RPM
- langextract: measured on both 1.6.0 and 1.7.0
- Python 3.14, Windows
- Raw PDF text: **159,629 characters**
- Entity classes wanted: **23**, split across 3 `extract()` calls to keep each JSON response short enough to avoid truncation (the workaround from #358)

At `max_char_buffer=3000`, chunking the raw document uniformly gives ~53 chunks per group, so **~159 API calls for a single filing**. On a 15 RPM free tier that is over ten minutes of wall clock, and the point of the project is to scan filings in bulk.

The answers are not spread evenly through these documents. The administrator, the management fee and the liquidation events each live under one known numbered clause. Most chunks contain no target entity at all, and each one still costs a request.

## Related to an issue?

Follow-up to #358, opened narrowly as requested when that issue was closed. This covers only the chunking item. The other items from #358 are either resolved (#509, #521) or closed (#520). Not asking to reopen anything.

## Possible solutions and alternatives

**What I do today, outside langextract.** A domain-specific pre-filter splits the regulation on its standard numbered clause headers (`7. TAXA DE ADMINISTRAÇÃO`, `26. EVENTOS DE LIQUIDAÇÃO`, …), keeps the first and last sections plus any section matching a keyword list, and drops the rest: **159,629 → 50,043 characters, a 69% reduction**, taking the run to ~16 chunks per group (~48 calls). Extraction quality on the kept sections is unchanged, because the dropped text genuinely does not contain the target entities.

It works, but it only works because I know this document family, and anyone applying langextract to long structured documents — contracts, filings, court records — has to rebuild the same thing.

**Preferred solution: a caller-supplied hook, evaluated per chunk before inference.**

```python
lx.extract(
    ...,
    chunk_filter=lambda chunk, prompt_description: bool(my_keywords & tokens(chunk)),
)
```

It receives the chunk text and the prompt description and returns whether to spend a call on it. langextract keeps full control of how chunks are formed, ordered and offset; the caller only decides what is worth sending. Source grounding is unaffected — skipped chunks simply produce no extractions, and surviving offsets still refer to the original document.

**Alternatives, if a hook is the wrong shape:**

1. A scoring variant returning a float, combined with a `max_chunks` budget — same effect, gives the library a say in the cutoff.
2. Expose the chunk list so callers can filter it and pass the survivors back in. No new inference-path behavior at all.
3. Relevance ranking inside langextract, driven by `prompt_description`. This was the original framing in #358 and I am explicitly *not* asking for it: it needs a design discussion and would be hard to make predictable across providers.

**What does not work: just using smaller chunks.** Worth stating, since it is the obvious first suggestion. My extraction retries a failed group at progressively smaller `max_char_buffer` (3000 → 1500 → 1000). Measuring four runs of this document on 2026-09-23, one run lost an entire entity group when all three sizes failed in turn with `ResolverParsingError`, while another run of the identical configuration succeeded at 3000 on the first attempt. Shrinking chunks costs strictly more calls without dependably recovering the group.

## Priority and timeline considerations

Nice to have, not time sensitive. There is a working local workaround, so nothing is blocked.

What raises it above cosmetic is that on a rate-limited tier the difference — ~159 versus ~48 calls per document — decides whether a corpus is processable at all. Alternative 2 (expose the chunk list) looks like the cheapest of the three if the priority is low.

## Additional context

The project is an open-source tool for investigative journalists to analyze fund regulations for transparency reporting: https://github.com/reichaves/langextract-fundos

Yes, happy to contribute. I can test a prototype against this document set and supply Brazilian Portuguese test cases, which the test corpus currently does not cover.
