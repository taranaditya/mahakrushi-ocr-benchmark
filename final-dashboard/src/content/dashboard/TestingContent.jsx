import React, { useEffect, useMemo, useState } from 'react';
import englishSample from '../assets/english-agriculture.png';
import hindiSample from '../assets/hindi-policy.png';
import marathiSample from '../assets/marathi-agriculture.png';
import './testing.css';

const API = window.location.hostname.endsWith('.ts.net')
  ? `${window.location.origin}/api`
  : 'http://127.0.0.1:8766';
const SAMPLES = [
  { id:'english', label:'English', note:'Agriculture recommendations', url:englishSample, file:'english-agriculture.png' },
  { id:'hindi', label:'हिन्दी', note:'Government policy letter', url:hindiSample, file:'hindi-policy.png' },
  { id:'marathi', label:'मराठी', note:'Agriculture research page', url:marathiSample, file:'marathi-agriculture.png' },
];
const kinds = { heading:'Heading', key_value:'Label / value', table_row:'Table row', list_item:'Numbered item', text:'Text' };
const DEMO_MODELS = [
  { id:'qwen2_5_vl_7b', label:'Qwen2.5-VL-7B', note:'Strong VLM' },
  { id:'easyocr', label:'EasyOCR', note:'Fast baseline' },
  { id:'glm_ocr', label:'GLM-OCR', note:'Lower-scoring comparison' },
  { id:'indic_ocr', label:'IndicOCR', note:'Indic-script specialist · CUDA' },
];

function readFile(file) {
  return new Promise((resolve,reject)=>{
    const reader=new FileReader();
    reader.onload=()=>resolve(reader.result);
    reader.onerror=()=>reject(new Error('Could not read this document.'));
    reader.readAsDataURL(file);
  });
}

export function TestingContent() {
  const [models,setModels]=useState([]), [model,setModel]=useState('qwen2_5_vl_7b');
  const [savedModels,setSavedModels]=useState({});
  const [connection,setConnection]=useState({state:'checking',connected:false});
  const [document,setDocument]=useState(null), [preview,setPreview]=useState(''), [sample,setSample]=useState('');
  const [busy,setBusy]=useState(false), [result,setResult]=useState(null), [error,setError]=useState('');
  const [page,setPage]=useState(1), [mode,setMode]=useState('structured');

  useEffect(()=>{
    let live=true;
    const check=async()=>{
      try {
        const [available,status,saved]=await Promise.all([
          fetch(`${API}/api/models`).then(r=>r.json()),
          fetch(`${API}/api/status?model=${encodeURIComponent(model)}`).then(r=>r.json()),
          fetch(`${API}/api/samples`).then(r=>r.json()),
        ]);
        if(live){setModels(available.models??[]);setConnection(status);setSavedModels(saved.modelsBySample??{});}
      } catch { if(live)setConnection({state:'offline',connected:false}); }
    };
    check();
    const timer=setInterval(check,20000);
    return ()=>{live=false;clearInterval(timer);};
  },[model]);

  useEffect(()=>{
    if(!document) return;
    const url=URL.createObjectURL(document);
    setPreview(url);
    return ()=>URL.revokeObjectURL(url);
  },[document]);

  const activePage=useMemo(()=>result?.pages?.find(item=>item.page===page),[result,page]);
  const selectedModel=models.find(item=>item.id===model);
  const savedAvailable=Boolean(sample&&savedModels[sample]?.includes(model));

  function selectFile(file,source='') {
    if(!file)return;
    if(file.size>40*1024*1024){setError('The document exceeds the 40 MB limit.');return;}
    setDocument(file);setSample(source);setResult(null);setError('');setPage(1);
  }

  async function selectSample(item) {
    try {
      const response=await fetch(item.url);
      if(!response.ok)throw new Error('Sample could not be loaded.');
      const blob=await response.blob();
      selectFile(new File([blob],item.file,{type:'image/png'}),item.id);
    } catch(err){setError(err.message);}
  }

  async function run() {
    if(!document||busy)return;
    setBusy(true);setResult(null);setError('');setPage(1);
    try {
      const file=await readFile(document);
      const response=await fetch(`${API}/api/ocr`,{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({model,file}),
      });
      const data=await response.json();
      if(!response.ok)throw new Error(data.error||`OCR failed (${response.status}).`);
      setResult(data);
      if(data.successful_pages===0)setError('The model returned no readable text for this document.');
    } catch(err){setError(err.message||'Could not complete OCR.');}
    finally{setBusy(false);}
  }

  async function showSaved() {
    if(!savedAvailable)return;
    setBusy(true);setError('');setResult(null);
    try {
      const response=await fetch(`${API}/api/sample?sample=${encodeURIComponent(sample)}&model=${encodeURIComponent(model)}`);
      const data=await response.json();
      if(!response.ok)throw new Error(data.error||'Saved result could not load.');
      setResult(data);setPage(1);
      if(!data.successful_pages)setError('This saved model run did not return readable text for the sample.');
    } catch(err){setError(err.message);}
    finally{setBusy(false);}
  }

  function downloadJson() {
    if(!result)return;
    const file=new Blob([JSON.stringify(result,null,2)],{type:'application/json'});
    const url=URL.createObjectURL(file);
    const link=window.document.createElement('a');
    link.href=url;link.download=`ocr-${result.model}-${Date.now()}.json`;link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  }

  return <article className="testing-page">
    <header className="testing-hero"><div className="ocr-overline"><span className="ocr-dot"/> MAHAKRUSHI / DOCUMENT LAB <span>ENGLISH · HINDI · MARATHI</span></div><p className="ocr-eyebrow">LIVE INFERENCE / DGX SPARK</p><h1>Test a <em>document.</em></h1><p>Choose a sample or upload a PDF or image. Pick a model, then inspect its extracted content beside the source.</p></header>
    <div className="testing-model-picker"><div className="testing-model-head"><div><label>OCR MODEL</label><small>{selectedModel?.benchmarkStatus?`Benchmark: ${selectedModel.benchmarkStatus}`:'Loading model status…'} · {connection.warm?'kept loaded on DGX':'checking warm service'}</small></div><div className={`testing-connection ${connection.state}`}><i/>{connection.warm?'MODEL WARM · DGX READY':connection.state==='ready'?'DGX GPU READY':connection.state==='queue'?'BENCHMARK QUEUE RUNNING':connection.state==='busy'?'DGX GPU BUSY':'DGX UNAVAILABLE'}</div></div><div className="testing-model-buttons">{DEMO_MODELS.map(item=><button key={item.id} className={model===item.id?'active':''} onClick={()=>{setModel(item.id);setConnection({state:'checking',connected:false});setResult(null);setError('');}} disabled={busy}><strong>{item.label}</strong><span>{item.note}</span></button>)}</div><div className="testing-actions">{savedAvailable&&<><span className="testing-cache-note">Saved benchmark output is optional; Run OCR always starts a fresh live DGX request.</span><button className="testing-saved" onClick={showSaved} disabled={busy}>VIEW SAVED OCR</button></>}<button className="testing-run" onClick={run} disabled={!document||busy||connection.state!=='ready'}>{busy?'PROCESSING ON DGX…':connection.state==='queue'?'WAITING FOR QUEUE':connection.state==='busy'?'GPU BUSY':'RUN LIVE OCR →'}</button></div></div>
    <div className="testing-workspace"><section className="testing-source"><div className="testing-panel-head"><div><span>01 / SOURCE DOCUMENT</span><h2>{document?.name??'Choose a document'}</h2></div>{document&&<span className="testing-file-size">{(document.size/1024/1024).toFixed(1)} MB</span>}</div>
      <div className="testing-samples">{SAMPLES.map(item=><button key={item.id} className={sample===item.id?'active':''} onClick={()=>selectSample(item)} disabled={busy}><strong>{item.label}</strong><span>{item.note}</span></button>)}</div>
      <label className="testing-upload">+ UPLOAD YOUR DOCUMENT <span>PDF, PNG, JPG, WebP or TIFF · up to 40 MB / 30 pages</span><input type="file" accept=".pdf,.png,.jpg,.jpeg,.webp,.tif,.tiff,application/pdf,image/*" onChange={e=>selectFile(e.target.files?.[0])} disabled={busy}/></label>
      <div className="testing-preview">{preview ? document?.type==='application/pdf'||document?.name.toLowerCase().endsWith('.pdf')?<iframe src={preview} title="Source PDF"/>:<img src={preview} alt="Source document preview"/>:<div className="testing-empty">Your document will appear here.</div>}</div>
    </section><section className="testing-output"><div className="testing-panel-head"><div><span>02 / OCR OUTPUT</span><h2>{result?'Extracted content':'Waiting for a document'}</h2></div>{result&&<span className="testing-file-size">{result.successful_pages}/{result.page_count} PAGES</span>}</div>
      {error&&<div className="testing-error">{error}</div>}
      {busy&&<div className="testing-running"><div className="testing-spinner"/><strong>Reading document on DGX Spark</strong><p>The selected model may take several minutes to load and process each page. Keep this tab open.</p></div>}
      {!busy&&!result&&!error&&<div className="testing-empty testing-output-empty"><span>OCR</span><p>Choose a source and click “Run OCR” to see its structured output here.</p></div>}
      {result&&<><div className="testing-result-bar"><div><span>MODEL</span><strong>{selectedModel?.name??result.model}</strong></div><div><span>{result.source==='saved_benchmark_prediction'?'SAVED BENCHMARK OCR':'LIVE DGX RESULT'}</span><strong>{result.successful_pages}/{result.page_count} pages read</strong></div><button onClick={downloadJson}>DOWNLOAD JSON ↗</button></div>
        <div className="testing-result-controls"><div>{result.pages.map(item=><button key={item.page} className={page===item.page?'active':''} onClick={()=>setPage(item.page)}>PAGE {item.page}</button>)}</div><div><button className={mode==='structured'?'active':''} onClick={()=>setMode('structured')}>STRUCTURED</button><button className={mode==='raw'?'active':''} onClick={()=>setMode('raw')}>RAW TEXT</button></div></div>
        <div className="testing-result-body">{activePage?.error&&<div className="testing-error">{activePage.error}</div>}{mode==='raw'?<pre className="testing-raw">{activePage?.text||'No text returned.'}</pre>:activePage?.blocks?.length?<div className="testing-blocks">{activePage.blocks.map((block,index)=><div key={index} className={`testing-block testing-${block.type}`}><small>{kinds[block.type]}</small>{block.type==='key_value'?<div className="testing-key-value"><strong>{block.label}</strong><span>{block.value}</span></div>:<p>{block.text}</p>}</div>)}</div>:<p className="testing-no-text">No structured content returned on this page.</p>}</div>
      </>}
    </section></div><footer className="ocr-footer"><span>MAHAKRUSHI · DOCUMENT TESTING</span><span>OCR TEXT SHOWN FOR REVIEW</span></footer>
  </article>;
}
