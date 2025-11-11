export type SupportedLanguage = string;

let currentStateName: string | null = null;
let currentLanguage: SupportedLanguage = 'English';

// Cache translations by `${lang}::${text}` to avoid re-calling backend
const translationCache = new Map<string, string>();
let mutationObserver: MutationObserver | null = null;
let observerActiveForLang: string | null = null;

export function getCurrentLanguage(): SupportedLanguage {
	return currentLanguage;
}

export function setCurrentLanguage(lang: SupportedLanguage) {
	currentLanguage = lang || 'English';
	// Re-arm observer when language changes
	setupMutationObserver();
}

export function setCurrentStateName(state: string | null) {
	currentStateName = state;
}

export async function fetchLanguagesForState(state: string): Promise<SupportedLanguage[]> {
	const params = new URLSearchParams({ state });
	const resp = await fetch(`http://localhost:5000/api/languages?${params.toString()}`);
	if (!resp.ok) {
		return ['English'];
	}
	const data = await resp.json();
	const arr = Array.isArray(data.languages) ? data.languages : [];
	return Array.from(new Set([...(arr as string[]), 'English']));
}

function collectTranslatable() {
	// Collect visible text nodes
	const textNodes: Text[] = [];
	const texts: string[] = [];

	const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
		acceptNode: (node: any) => {
			const text = (node.textContent || '').trim();
			if (!text) return NodeFilter.FILTER_REJECT;
			const parent = node.parentElement as HTMLElement | null;
			if (!parent) return NodeFilter.FILTER_REJECT;
			const tag = parent.tagName.toLowerCase();
			if (['script', 'style', 'noscript'].includes(tag)) return NodeFilter.FILTER_REJECT;
			if (/^[\d\s\W]+$/.test(text)) return NodeFilter.FILTER_REJECT;
			// Skip elements explicitly marked to ignore
			if (parent.hasAttribute('data-i18n-ignore')) return NodeFilter.FILTER_REJECT;
			return NodeFilter.FILTER_ACCEPT;
		},
	} as any);

	let n: Node | null = walker.nextNode();
	while (n) {
		const t = (n as Text);
		const content = (t.textContent || '').trim();
		textNodes.push(t);
		texts.push(content);
		n = walker.nextNode();
	}

	// Attributes to translate
	const attrTargets: Array<{ el: HTMLElement, attr: string, text: string }> = [];
	const ATTRS = ['placeholder', 'aria-label', 'title'];
	const allElements = Array.from(document.body.querySelectorAll<HTMLElement>('*'));
	for (const el of allElements) {
		if (el.hasAttribute('data-i18n-ignore')) continue;
		for (const attr of ATTRS) {
			if (el.hasAttribute(attr)) {
				const raw = (el.getAttribute(attr) || '').trim();
				if (raw && !/^[\d\s\W]+$/.test(raw)) {
					attrTargets.push({ el, attr, text: raw });
				}
			}
		}
	}

	return { textNodes, texts, attrTargets };
}

async function translateBatch(texts: string[], targetLanguage: SupportedLanguage): Promise<string[]> {
	if (texts.length === 0) return [];

	// Try cache
	const results: string[] = new Array(texts.length);
	const missing: { index: number, text: string }[] = [];
	for (let i = 0; i < texts.length; i++) {
		const key = `${targetLanguage}::${texts[i]}`;
		const cached = translationCache.get(key);
		if (cached !== undefined) {
			results[i] = cached;
		} else {
			missing.push({ index: i, text: texts[i] });
		}
	}

	if (missing.length === 0) return results;

	// Chunk missing to keep payload bounded
	const chunkSize = 40;
	for (let start = 0; start < missing.length; start += chunkSize) {
		const slice = missing.slice(start, start + chunkSize);
		const payload = slice.map(x => x.text);
		try {
			const resp = await fetch('http://localhost:5000/api/translate', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ texts: payload, targetLanguage }),
			});
			if (!resp.ok) continue;
			const data = await resp.json();
			const translations: string[] = Array.isArray(data.translations) ? data.translations : [];
			for (let j = 0; j < slice.length; j++) {
				const globalIndex = slice[j].index;
				const translated = translations[j] ?? payload[j];
				results[globalIndex] = translated;
				translationCache.set(`${targetLanguage}::${slice[j].text}`, translated);
			}
		} catch {
			for (let j = 0; j < slice.length; j++) {
				const globalIndex = slice[j].index;
				results[globalIndex] = payload[j];
			}
		}
	}

	return results;
}

export async function translatePage(targetLanguage: SupportedLanguage): Promise<void> {
	// If English, do nothing (original text remains via React rendering)
	if (!targetLanguage || targetLanguage.toLowerCase() === 'english') {
		observerActiveForLang = 'English';
		return;
	}

	const { textNodes, texts, attrTargets } = collectTranslatable();

	// Translate text nodes
	const translatedTexts = await translateBatch(texts, targetLanguage);
	for (let i = 0; i < textNodes.length; i++) {
		if (translatedTexts[i]) {
			textNodes[i].textContent = translatedTexts[i];
		}
	}

	// Translate attributes
	const attrTexts = attrTargets.map(a => a.text);
	const translatedAttrs = await translateBatch(attrTexts, targetLanguage);
	for (let i = 0; i < attrTargets.length; i++) {
		const tgt = attrTargets[i];
		const val = translatedAttrs[i];
		if (val && tgt.el.hasAttribute(tgt.attr)) {
			tgt.el.setAttribute(tgt.attr, val);
		}
	}

	// Ensure we keep translating after React updates
	setupMutationObserver();
}

function setupMutationObserver() {
	const lang = (currentLanguage || 'English');
	if (lang.toLowerCase() === 'english') {
		if (mutationObserver) {
			mutationObserver.disconnect();
			mutationObserver = null;
		}
		observerActiveForLang = 'English';
		return;
	}

	// If already observing for this language, skip
	if (observerActiveForLang === lang && mutationObserver) return;

	if (mutationObserver) {
		mutationObserver.disconnect();
	}

	mutationObserver = new MutationObserver(async (mutations) => {
		// Batch re-translate only if nodes added or text changed
		let needs = false;
		for (const m of mutations) {
			if (m.type === 'childList' && (m.addedNodes.length > 0 || m.removedNodes.length > 0)) {
				needs = true;
				break;
			}
			if (m.type === 'characterData') {
				needs = true;
				break;
			}
			if (m.type === 'attributes') {
				needs = true;
				break;
			}
		}
		if (needs) {
			await translatePage(lang);
		}
	});

	mutationObserver.observe(document.body, {
		childList: true,
		subtree: true,
		characterData: true,
		attributes: true,
		attributeFilter: ['placeholder', 'aria-label', 'title'],
	});

	observerActiveForLang = lang;
}


