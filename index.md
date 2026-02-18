---
layout: page
title: "Protein & Artificial Intelligence"
description: "Spring 2026 · KAIST"
---

{% assign course = site.data.course %}

**{{ course.semester }}** · {{ course.institution }}{% if course.co_instructors %} · Co-taught with {{ course.co_instructors | join: " and " }}{% endif %}

{{ course.description }}

**Prerequisites:** {{ course.prerequisites }}

**Textbooks:** Our lecture notes are self-contained, but students who want deeper explanations of the deep learning foundations may find these two open-access textbooks helpful:
- Zhang et al., [*Dive into Deep Learning*](https://d2l.ai/) (CC BY-SA 4.0) — hands-on, code-first introduction to deep learning with PyTorch.
- Prince, [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/) (CC BY-NC-ND) — conceptual and mathematical treatment with excellent figures.

{% assign notes = site.lectures | sort: "lecture_number" %}
{% assign preliminary = notes | where: "preliminary", true %}
{% assign lectures = notes | where_exp: "item", "item.preliminary != true" %}

{% if preliminary.size > 0 %}
### Preliminary Notes
<p class="post-description" style="margin-bottom: 0.5em;">{{ course.preliminary_description }}</p>

<ol start="0">
{% for note in preliminary %}
  <li value="{{ note.lecture_number }}"><a href="{{ note.url | relative_url }}">{{ note.title }}</a> — {{ note.description }}</li>
{% endfor %}
</ol>
{% endif %}

{% if lectures.size > 0 %}
### Lectures
<p class="post-description" style="margin-bottom: 0.5em;">{{ course.lectures_description }}</p>

<ol>
{% for lecture in lectures %}
  <li><a href="{{ lecture.url | relative_url }}">{{ lecture.title }}</a> — {{ lecture.date | date: "%b %d" }}. {{ lecture.description }}</li>
{% endfor %}
</ol>
{% endif %}

{% if course.references.size > 0 %}
#### Key References

{% for ref in course.references %}- {{ ref }}
{% endfor %}
{% endif %}
