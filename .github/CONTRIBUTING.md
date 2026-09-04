# Contribution Guidelines and AI Policy
We welcome pull requests and are happy to see the
community adopt OpenEquivariance. Here are some ground rules.

## Use of Large Language Models
Large language models are useful tools that we use as maintainers.
That said, LLMs often code in a counterproductive style and shift the
balance of work disproportionately towards reviewers.
We adopt [LLVM's AI Tool Use Policy](https://llvm.org/docs/AIToolPolicy.html)
and expect contributors to adhere to it. The document is
worth reading in its entirety, but three salient points are:

1. All code must be read, reviewed, and understood by the contributor before submitting
   a pull request.

2. Contributions should be *non-extractive*. A non-extractive contribution,
   as defined by the document, is worth more to the project than the
   time it takes to review it. For example, features that support
   new models, ease end-user workflows, or fix bugs are welcome.

3. Contributors are responsible for ensuring that any added code
   does not violate copyright protections.

## Coding Style
To these guidelines, we add a few more that are relevant for those
developing with AI assistance.

**Do not include comments in your code (with limited exceptions).**

This is an extreme position, but LLMs produce bulky comment blocks
that restate logic that a quick scan of the code should reveal. Worse, a comment
that falls out of sync with the code it describes can dupe both an LLM
and a human (see [The Pragmatic Programmer](https://en.wikipedia.org/wiki/The_Pragmatic_Programmer)).

Your code should be self-documenting with a few exceptions: comments intended
for Sphinx documentation are okay, since they are for library clients.
In extremely limited circumstances, comments for complicated algorithms
are fine when they explain a design choice that the code does not make clear.
All comments must be human-written.

**Minimize spurious diffs.**

These are changes like removing whitespace, changing indentation,
etc., that are unrelated to the logic your PR contributes and create
noise for reviewers.

