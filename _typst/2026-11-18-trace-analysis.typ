// Metadata is consumed by tools/build_typst_posts.py.
#let title = "Trace Analysis"
#let subtitle = "What we can see from the trace of agents"
#let author = "TBD"
#let published = datetime(year: 2026, month: 11, day: 18)

#metadata((
  title: title,
  date: "2026-11-18",
  excerpt: subtitle,
  draft: true,
)) <blog-meta>

#set document(title: title, author: author)
#set text(size: 11pt, lang: "en")
#set par(justify: true, leading: 0.7em)
#set heading(numbering: none)
#show link: set text(fill: rgb("2563eb"))

// Jekyll supplies the web title and date. Keep the cover only for PDF export.
#let is-html = sys.inputs.at("target", default: "pdf") == "html"
#if not is-html [
  #set page(paper: "a4", margin: (x: 2.4cm, y: 2.6cm))
  #align(center)[
    #text(size: 26pt, weight: "bold")[#title]

    #v(0.4em)
    #text(size: 13pt, fill: rgb("525252"))[#subtitle]

    #v(1em)
    #text(size: 10pt, fill: rgb("737373"))[
      #author · #published.display("[month repr:long] [day], [year]")
    ]
  ]
  #v(2em)
]

= Introduction

Agents create large amount of trajectories, and extracting knowledge from them is difficult. But generally, we look forward to see how agents deal with the tasks at hand, and what's the difference between them and human experts.

To do this, several evaluation methods have been developed.

= References

+ Liliang Weng, 2026. #link("https://lilianweng.github.io/posts/2026-07-04-harness/", [Lil's Blog])
