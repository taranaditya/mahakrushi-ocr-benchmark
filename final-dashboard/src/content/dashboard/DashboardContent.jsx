import React, { useMemo, useState } from 'react';
import { DataComponent, useDataApp, useDashboardTabs } from '../../data-app-public.jsx';
import { TestingContent } from './TestingContent.jsx';
import './dashboard.css';

const pct = x => Number.isFinite(x) ? `${(100*x).toFixed(1)}%` : '—';
const sec = x => Number.isFinite(x) ? `${x.toFixed(1)}s` : '—';
const langs = { all:'All languages', en:'English', hi:'Hindi', mr:'Marathi' };
const field = { cer:'cer_all_pages', cerNorm:'cer_normalized_ocrd_all_pages', wer:'wer_all_pages', exact:'normalized_exact_match_rate', mean:'mean_latency_seconds', pages:'pages', attempted:'attempted_pages', successful:'successful_pages', failed:'failed_or_missing_pages', editorial:'editorial_bracket_pages' };
const value = (r,l,k) => l==='all' ? r[k] : r[l]?.[field[k]];
const modelParameters = {
  qwen2_5_vl_7b:'7B', easyocr:'Pipeline', chitrapathak_2:'9B', nemotron_parse_2:'1B',
  qianfan_ocr:'4B', jina_ocr_v1:'1B', deepseek_ocr:'3B', indicphotoocr:'Pipeline',
  unlimited_ocr:'3B', chandra_ocr_2:'9B', paddleocr_vl_1_6:'0.9B', paddleocr_vl_v1:'0.9B',
  deepseek_ocr_2:'3B', glm_ocr:'0.9B', surya_ocr_2:'Pipeline', ovisocr2:'0.9B',
  qwen3_vl_8b:'8B', ministral_3b:'3B', ministral_8b:'8B', sarvam_1_vl_4b:'4B', qwen3_5_9b:'9B', indic_ocr:'0.8B + 33M'
};

export function DashboardContent() {
  const { activeTabId } = useDashboardTabs([{ id:'results', label:'Results' }, { id:'testing', label:'Testing' }]);
  const { reviewedRows } = useDataApp();
  const rows = reviewedRows('scorecards');
  const [lang,setLang]=useState('all'), [scope,setScope]=useState('complete'), [sort,setSort]=useState('cer'), [selected,setSelected]=useState(null);
  const complete=rows.filter(r=>r.status==='complete');
  const shown=useMemo(()=>rows.filter(r=>scope==='all'||r.status===scope).sort((a,b)=>
    (a.status==='complete'?0:1)-(b.status==='complete'?0:1)||(value(a,lang,sort)??Infinity)-(value(b,lang,sort)??Infinity)),[rows,lang,scope,sort]);
  const leader=[...complete].sort((a,b)=>(value(a,lang,'cer')??Infinity)-(value(b,lang,'cer')??Infinity))[0];
  const detail=rows.find(r=>r.id===selected)??leader;
  const successes=rows.reduce((n,r)=>n+r.successful,0);
  if (activeTabId === 'testing') return <TestingContent/>;
  return <article className="ocr-page">
    <header className="ocr-hero"><div className="ocr-overline"><span className="ocr-dot"/> MAHAKRUSHI / FINAL OCR TEST <span>DGX SPARK · 31 DOCUMENT PAGES</span></div>
      <p className="ocr-eyebrow">MODEL EVALUATION / SEPTEMBER 2026</p><h1>OCR <em>benchmark.</em></h1><p className="ocr-intro">A clear view of transcription quality, speed, and run reliability across English, Hindi, and Marathi policy documents.</p>
      <div className="ocr-notice"><strong>PROVISIONAL EVIDENCE</strong><span>References have not been independently visually approved. This measures page transcription, not form-field extraction. Only complete 31/31 runs appear in the ranked default view.</span></div>
    </header>
    <section className="ocr-kpis" aria-label="Summary"><div><label>MODELS SCORED</label><strong>{rows.length}<small> / {complete.length} complete</small></strong><p>{rows.filter(r=>r.status==='partial').length} partial · {rows.filter(r=>r.status==='failed').length} failed</p></div><div><label>LOWEST CHARACTER ERROR</label><strong>{leader?pct(value(leader,lang,'cer')):'—'}</strong><p>{leader?.model??'No complete run'}</p></div><div><label>DOCUMENTS PROCESSED</label><strong>{successes}<small> / {rows.length*31}</small></strong><p>Successful page-model attempts</p></div><div><label>LANGUAGE MIX</label><strong>3<small> languages</small></strong><p>English 6 · Hindi 3 · Marathi 22</p></div></section>
    <section className="ocr-section"><div className="ocr-section-head"><div><small>01 / LEADERBOARD</small><h2>Model comparison</h2><p>Lower error and latency are better. Higher exact match is better.</p></div><span>{shown.length} MODELS SHOWN</span></div>
      <div className="ocr-controls"><div className="ocr-tabs">{Object.entries(langs).map(([k,n])=><button key={k} className={lang===k?'active':''} onClick={()=>setLang(k)}>{n}</button>)}</div><div className="ocr-selects"><label>RUNS <select value={scope} onChange={e=>setScope(e.target.value)}><option value="complete">Complete only</option><option value="all">All statuses</option><option value="partial">Partial</option><option value="failed">Failed</option></select></label><label>SORT <select value={sort} onChange={e=>setSort(e.target.value)}><option value="cer">Character error</option><option value="cerNorm">Normalized CER (OCR-D)</option><option value="wer">Word error</option><option value="mean">Mean latency</option></select></label></div></div>
      <DataComponent id="ocr-leaderboard" queryId="scorecards" title="Final OCR scorecards" displayRows={shown} sourceRows={rows} variant="plain"><div className="ocr-scroll" data-reviewed-rows><table className="ocr-table"><thead><tr><th>RANK / MODEL</th><th>PARAMS</th><th>STATUS</th><th>ATTEMPTED</th><th>SUCCESS</th><th>FAILED</th><th title="Strict CER: grapheme-level Levenshtein edit count divided by reference grapheme clusters. NFC; whitespace, punctuation and case are retained. Can exceed 100% when output contains many insertions.">CER ALL ↓</th><th title="OCR-D bounded CER-N: grapheme-level Levenshtein errors divided by errors plus correctly aligned graphemes. NFC; case, punctuation, whitespace and Indic marks are retained. Rankings still use strict reference-length CER.">CER-N (OCR-D) ↓</th><th>WER ↓</th><th>EXACT ↑</th><th>MEAN ↓</th><th>REVIEW MARKS</th></tr></thead><tbody>{shown.map((r,i)=><tr key={r.id} className={detail?.id===r.id?'selected':''} onClick={()=>setSelected(r.id)}><td className="ocr-model"><b>{r.status==='complete'?String(i+1).padStart(2,'0'):'—'}</b><span>{r.model}<small>{r.runtime?.replaceAll('_',' ')}</small></span></td><td><strong>{modelParameters[r.id]??'—'}</strong></td><td><mark className={`ocr-status ${r.status}`}>{r.status}</mark></td><td>{value(r,lang,'attempted')??0}/{value(r,lang,'pages')??0}</td><td>{value(r,lang,'successful')??0}</td><td>{value(r,lang,'failed')??0}</td><td className="ocr-highlight">{pct(value(r,lang,'cer'))}</td><td>{pct(value(r,lang,'cerNorm'))}</td><td>{pct(value(r,lang,'wer'))}</td><td>{pct(value(r,lang,'exact'))}</td><td>{sec(value(r,lang,'mean'))}</td><td>{value(r,lang,'editorial')??'—'}</td></tr>)}</tbody></table></div><p className="ocr-metric-note">Strict CER counts grapheme-level insertions, substitutions and deletions per reference grapheme. CER-N uses OCR-D’s bounded ratio: errors ÷ (errors + correctly aligned graphemes), so it stays between 0% and 100%. Model ranks use strict CER.</p></DataComponent>
    </section>
    <section className="ocr-section ocr-detail"><div className="ocr-section-head"><div><small>02 / INSPECTION</small><h2>{detail?.model??'Model detail'}</h2><p>Selected model across the three language groups and full dataset.</p></div><mark className={`ocr-status ${detail?.status}`}>{detail?.status}</mark></div>
      {detail&&<DataComponent id="ocr-language-detail" queryId="scorecards" title="Language breakdown" displayRows={[detail]} sourceRows={rows} variant="plain"><div className="ocr-scroll" data-reviewed-rows><table className="ocr-table"><thead><tr><th>LANGUAGE</th><th>PAGES</th><th>ATTEMPTED</th><th>SUCCESS</th><th>FAILED</th><th title="Strict CER: grapheme-level Levenshtein edits per reference grapheme; NFC, whitespace and punctuation retained.">CER ALL ↓</th><th title="OCR-D bounded CER-N over grapheme clusters; retains Indic marks, punctuation, case and whitespace.">CER-N (OCR-D) ↓</th><th>WER ↓</th><th>EXACT ↑</th><th>MEAN ↓</th><th>REVIEW MARKS</th></tr></thead><tbody>{Object.entries(langs).map(([k,n])=><tr key={k}><td>{n}</td><td>{value(detail,k,'pages')??'—'}</td><td>{value(detail,k,'attempted')??'—'}</td><td>{value(detail,k,'successful')??'—'}</td><td>{value(detail,k,'failed')??'—'}</td><td className="ocr-highlight">{pct(value(detail,k,'cer'))}</td><td>{pct(value(detail,k,'cerNorm'))}</td><td>{pct(value(detail,k,'wer'))}</td><td>{pct(value(detail,k,'exact'))}</td><td>{sec(value(detail,k,'mean'))}</td><td>{value(detail,k,'editorial')??'—'}</td></tr>)}</tbody></table></div></DataComponent>}
    </section><footer className="ocr-footer"><span>MAHAKRUSHI · OCR EVALUATION</span><span>Snapshot from DGX scorecards · 25 SEP 2026</span></footer>
  </article>;
}
