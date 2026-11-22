# Typst posts

Write posts as `YYYY-MM-DD-slug.typ` in this directory. The source remains Typst; `tools/build_typst_posts.py` generates a Jekyll HTML post and checks the PDF export. Generated files in `_posts/typst/` and `files/typst/` should not be edited or committed.

> Typst 0.15.1's HTML export is experimental and incomplete. This integration is an opt-in authoring workflow, not native Jekyll support. Check new layouts before publishing and keep `draft: true` for unfinished work. See <https://typst.app/docs/reference/html/>.

## Metadata

Each source must contain exactly one labelled metadata dictionary:

```typst
#metadata((
  title: "A readable title",
  date: "2026-11-19",
  excerpt: "A short summary for the home page.",
  draft: true,
  tags: ["Optional", "Tags"],
)) <blog-meta>
```

The date must match the filename. Jekyll supplies the page title, date, navigation, and reading layout. Do not add a second title block for HTML. A PDF-only cover can be wrapped in:

```typst
#let is-html = sys.inputs.at("target", default: "pdf") == "html"
#if not is-html [
  // PDF title and page settings here.
]
```

Use normal Typst headings, links, math, code, and footnotes for the body. The build preserves Typst's MathML and its required math styles, and disables MathJax on generated pages. Existing Markdown posts are unchanged.

## Build

With Typst 0.15.1 or a compatible newer version:

```sh
python3 tools/build_typst_posts.py
```

Or select a compiler explicitly:

```sh
TYPST=/path/to/typst python3 tools/build_typst_posts.py
```

Then run the normal Jekyll build. Draft HTML is generated under `_posts/typst/` with `published: false`, so it does not appear on the home page or production site. Its PDF is validated in a temporary directory but not copied into public files.

After reviewing the rendered article, set `draft: false` and rebuild. This publishes the HTML post, a PDF under `files/typst/`, and a downloadable copy of the Typst source. Removing or renaming a source currently requires removing its corresponding generated files too.
