import React, { useEffect, useState } from 'react';
import { fetchLanguagesForState, setCurrentLanguage, setCurrentStateName, translatePage } from '@/lib/language';

const indianStates = [
	'Andhra Pradesh','Arunachal Pradesh','Assam','Bihar','Chhattisgarh','Goa','Gujarat','Haryana','Himachal Pradesh','Jharkhand','Karnataka','Kerala','Madhya Pradesh','Maharashtra','Manipur','Meghalaya','Mizoram','Nagaland','Odisha','Punjab','Rajasthan','Sikkim','Tamil Nadu','Telangana','Tripura','Uttar Pradesh','Uttarakhand','West Bengal','Delhi','Jammu and Kashmir','Ladakh'
];

export const LanguageSelector: React.FC = () => {
	const [stateName, setStateName] = useState<string>('Delhi');
	const [languages, setLanguages] = useState<string[]>(['English']);
	const [language, setLanguage] = useState<string>('English');
	const [pendingLanguage, setPendingLanguage] = useState<string>('English');
	const [loading, setLoading] = useState<boolean>(false);

	useEffect(() => {
		// initial fetch
		let ignore = false;
		(async () => {
			setLoading(true);
			try {
				const langs = await fetchLanguagesForState(stateName);
				if (!ignore) {
					setLanguages(langs);
				}
			} finally {
				if (!ignore) setLoading(false);
			}
		})();
		return () => { ignore = true };
	}, []);

	const onStateChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
		const val = e.target.value;
		setStateName(val);
		setCurrentStateName(val);
		setLoading(true);
		try {
			const langs = await fetchLanguagesForState(val);
			setLanguages(langs);
			// Keep English selected unless user changes
			if (!langs.includes(language)) {
				setLanguage('English');
				setPendingLanguage('English');
			}
		} finally {
			setLoading(false);
		}
	};

	const onLanguageChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
		const val = e.target.value;
		setPendingLanguage(val);
	};

	const applyLanguage = async () => {
		setLanguage(pendingLanguage);
		setCurrentLanguage(pendingLanguage);
		await translatePage(pendingLanguage);
	};

	return (
		<div className="flex items-center gap-2">
			<select
				value={stateName}
				onChange={onStateChange}
				aria-label="Select State"
				className="px-2 py-1 rounded-md border border-border bg-transparent text-white text-sm"
			>
				{indianStates.map(s => (
					<option key={s} value={s} className="bg-card">{s}</option>
				))}
			</select>
			<select
				value={pendingLanguage}
				onChange={onLanguageChange}
				aria-label="Select Language"
				className="px-2 py-1 rounded-md border border-border bg-transparent text-white text-sm"
				disabled={loading}
			>
				{languages.map(l => (
					<option key={l} value={l} className="bg-card">{l}</option>
				))}
			</select>
			<button
				onClick={applyLanguage}
				disabled={loading}
				className="px-2 py-1 rounded-md border border-border bg-transparent hover:bg-accent text-white text-sm disabled:opacity-60"
				aria-label="Apply Language"
			>
				Apply
			</button>
		</div>
	);
};

export default LanguageSelector;


