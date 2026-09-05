#!/usr/bin/env python3
"""Bounded pinned-model/backend/padding probe; source-only, no database or remote model calls."""
import json
import time
from pathlib import Path
from uuid import UUID
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification
from rag_kb.adapters.local_reranker import LocalMiniLmTokenizer, build_local_rerank_windows
from rag_kb.domain import RerankDocument
from tools.evaluate_auto_qa_retrieval import _evaluation_cases, DEFAULT_CORPUS


def main():
    OUT=Path(".runtime/evaluations/minilm-source-20260905")
    REF=Path('.runtime/model-assets/minilm-reference-1427fd6')
    from hashlib import sha256
    from tools.evaluate_minilm_source import REFERENCE_SHA256, REVISION
    from rag_kb.adapters.local_reranker_artifacts import verify_local_reranker_artifacts
    from tools.evaluate_auto_qa_reuse import _write
    if sha256((REF/'onnx/model.onnx').read_bytes()).hexdigest()!=REFERENCE_SHA256:
        raise RuntimeError('Float reference hash mismatch')
    if sha256((REF/'model.safetensors').read_bytes()).hexdigest()!='5daeca2481a76b5976a2bdc32f0a78532b6716da4f8cd3ff59460ef8d2f359b4':
        raise RuntimeError('Native reference hash mismatch')
    if json.loads((REF/'reference-manifest.json').read_text())['revision'] != REVISION:
        raise RuntimeError('Wrong model revision')
    OLD=Path('.runtime/model-assets/local-reranker')
    verify_local_reranker_artifacts(OLD, Path('config/local-reranker-artifacts-v1.json'))
    diagnosis=json.loads(Path('.runtime/evaluations/auto-qa-diagnosis-20260905/analysis.json').read_text())
    sources=json.loads(Path('.runtime/evaluations/auto-qa-diagnosis-20260905/sources.json').read_text())['chunks']
    cases={c['evaluation_case_id']:c for c in _evaluation_cases(DEFAULT_CORPUS)}
    selected=['cfqa-101','para-cfqa-89','financebench_id_00563','financebench_id_01091',
              'financebench_id_00678','tatqa-05b670d3-5b19-438c-873f-9bf6de29c69e']
    tokenizer=LocalMiniLmTokenizer.load(OLD)
    windows=[]; pairs=[]
    for cid in selected:
        case=cases[cid]; audit=next(a for a in diagnosis['audit'] if a['case_id']==cid)
        old_top=next(x for x in diagnosis['arms']['model_raw_source']['cases'] if x['case_id']==cid)['final_ids'][0]
        classic_top=next(x for x in diagnosis['arms']['classic_source']['cases'] if x['case_id']==cid)['final_ids'][0]
        for h in audit['candidates']:
            if not h['source'] or not (h['historical_label_match'] or h['id'] in {old_top,classic_top}): continue
            c=sources[h['id']]
            doc=RerankDocument(index_chunk_id=UUID(h['id']),text=c['content'],hierarchy=c['hierarchy'],modality=c['modality'])
            ws=build_local_rerank_windows(tokenizer,case['question'],(doc,))
            pairs.append({'case_id':cid,'id':h['id'],'label':h['historical_label_match'],
                          'text':c['content'],'query':case['question'],'cached_logit':h['scores']['source']['raw_logit'],
                          'windows':list(range(len(windows),len(windows)+len(ws)))})
            windows.extend(ws)
    if len(pairs) != 21 or len(windows) != 27:
        raise RuntimeError('Frozen probe membership changed')
    print(json.dumps({'pairs':len(pairs),'windows':len(windows)}),flush=True)

    def session(path):
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
        opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])

    def score(model,batch_size=8,pad_to=None,native=False):
        values=[]
        for start in range(0,len(windows),batch_size):
            batch=windows[start:start+batch_size];size=pad_to or max(len(w.input_ids) for w in batch)
            ids=np.full((len(batch),size),tokenizer.pad_token_id,dtype=np.int64);mask=np.zeros_like(ids)
            for row,w in enumerate(batch):
                ids[row,:len(w.input_ids)]=w.input_ids;mask[row,:len(w.input_ids)]=1
            if native:
                with torch.inference_mode():
                    output=model(input_ids=torch.from_numpy(ids),attention_mask=torch.from_numpy(mask)).logits.numpy()
            else: output=model.run(None,{'input_ids':ids,'attention_mask':mask})[0]
            logits = np.asarray(output).reshape(-1)
            if len(logits) != len(batch) or not np.isfinite(logits).all():
                raise RuntimeError('Invalid model output')
            values.extend(logits.tolist())
        return values

    results={}
    for name,path in [('int8',OLD/'onnx/model_qint8_arm64.onnx'),('onnx_fp32',REF/'onnx/model.onnx')]:
        model=session(path)
        for batch,pad in [(8,None),(1,None),(8,512)]:
            tag=f'{name}_b{batch}_pad{pad}';start=time.perf_counter()
            results[tag]=score(model,batch,pad)
            print(json.dumps({'arm':tag,'seconds':time.perf_counter()-start}),flush=True)
        del model
    torch.set_num_threads(2)
    model=AutoModelForSequenceClassification.from_pretrained(REF,local_files_only=True,trust_remote_code=False,
                                                           use_safetensors=True,attn_implementation='eager').eval()
    start=time.perf_counter();results['torch_fp32']=score(model,native=True)
    print(json.dumps({'arm':'torch_fp32','seconds':time.perf_counter()-start}),flush=True)
    for p in pairs:
        p['logits']={name:max(values[i] for i in p['windows']) for name,values in results.items()}
        print(json.dumps({k:v for k,v in p.items() if k not in {'text','query','windows'}},ensure_ascii=False),flush=True)
    def difference(left, right):
        return np.abs(np.asarray(results[left]) - np.asarray(results[right]))

    summary = {
        "pairs": len(pairs), "windows": len(windows),
        "onnx_vs_native_max_abs": float(difference("onnx_fp32_b8_padNone", "torch_fp32").max()),
        "onnx_padding_max_abs": float(difference("onnx_fp32_b8_padNone", "onnx_fp32_b8_pad512").max()),
        "onnx_batch_max_abs": float(difference("onnx_fp32_b8_padNone", "onnx_fp32_b1_padNone").max()),
        "int8_vs_float_mean_abs": float(difference("int8_b8_padNone", "torch_fp32").mean()),
        "int8_vs_float_max_abs": float(difference("int8_b8_padNone", "torch_fp32").max()),
        "int8_batch_max_abs": float(difference("int8_b8_padNone", "int8_b1_padNone").max()),
    }
    out = {"pairs": pairs, "window_scores": results, "summary": summary,
           "reference_revision": REVISION, "onnx_sha256": REFERENCE_SHA256,
           "qa_generation_calls": 0, "remote_model_calls": 0, "source_only": True}
    _write(OUT / "probe.json", out)
    _write(OUT / "probe-summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
