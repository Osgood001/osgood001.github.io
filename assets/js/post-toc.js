(() => {
  const content = document.querySelector('.post-article .page__content');
  const toc = document.querySelector('.page__toc');
  const navigation = document.querySelector('[data-post-toc]');

  if (!content || !toc || !navigation) return;

  const headings = [...content.querySelectorAll('h2, h3')].filter(
    heading => heading.textContent.trim() && !heading.closest('.toc')
  );

  if (!headings.length) {
    toc.hidden = true;
    return;
  }

  const slug = (text, index) => {
    const normalized = text
      .trim()
      .toLowerCase()
      .normalize('NFKD')
      .replace(/[\u0300-\u036f]/g, '')
      .replace(/[^\p{Letter}\p{Number}]+/gu, '-')
      .replace(/^-+|-+$/g, '');
    return normalized || `section-${index + 1}`;
  };

  const usedIds = new Set();
  const list = document.createElement('ul');
  list.className = 'toc__menu';
  let sectionItem = null;
  let subsectionList = null;

  headings.forEach((heading, index) => {
    let id = heading.id || slug(heading.textContent, index);
    const baseId = id;
    let suffix = 2;
    while (usedIds.has(id) || document.getElementById(id) !== heading && document.getElementById(id)) {
      id = `${baseId}-${suffix++}`;
    }

    heading.id = id;
    usedIds.add(id);

    heading.classList.add('page__section');
    heading.addEventListener('click', () => {
      window.location.hash = id;
    });

    const item = document.createElement('li');
    const link = document.createElement('a');
    link.href = `#${id}`;
    link.textContent = heading.textContent.trim();
    item.append(link);

    if (heading.tagName === 'H3' && sectionItem) {
      item.className = 'toc__subitem';
      if (!subsectionList) {
        subsectionList = document.createElement('ul');
        sectionItem.append(subsectionList);
      }
      subsectionList.append(item);
    } else {
      list.append(item);
      sectionItem = heading.tagName === 'H2' ? item : null;
      subsectionList = null;
    }
  });

  navigation.append(list);

  const wideLayout = window.matchMedia('(min-width: 1200px)');
  const syncTocState = () => {
    toc.open = wideLayout.matches;
  };
  syncTocState();
  wideLayout.addEventListener('change', syncTocState);

  if (window.location.hash) {
    const heading = document.getElementById(window.location.hash.slice(1));
    if (heading) heading.scrollIntoView();
  }
})();
