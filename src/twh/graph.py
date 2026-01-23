#!/usr/bin/env python3

"""
Here’s a no-fuss way to turn your task list (e.g., Taskwarrior export) into a clean Mermaid flowchart and a CSV you can paste straight into Tana.

What this does (in plain English)

Reads tasks (JSON on stdin), dedupes edges, and collapses trivial chains into single arrows for a tidy Mermaid diagram.

Emits tasks.csv with columns Title, Body, Tags, UUID, DependsOn so you can copy‑paste into Google Sheets → paste into Tana to recreate links or turn them into references.

No TUI, no extra tooling—just Python.

Usage
-----

    task export | python3 task2mermaid_tana.py > graph.mmd

Files produced: graph.mmd (Mermaid) and tasks.csv (for Tana import).

Quick tips

View Mermaid: open graph.mmd in a Mermaid live editor or VS Code Mermaid preview.

Tana import: open tasks.csv in Google Sheets; copy all; paste into Tana. UUID and DependsOn let you rebuild dependency links.

Windows/WSL: this is editor/browser‑based (no Graphviz needed). If you want DOT rendering, install Graphviz (WSL often smoother).

If you want, I can also add a tiny flag to filter by +PROJECT or collapse/expand by tag.
"""
import sys,json,csv
from collections import defaultdict

def read_tasks():
    data = json.load(sys.stdin)
    return data if isinstance(data,list) else [data]

def deps_list(f):
    if not f: return []
    if isinstance(f,list): return f
    return [x.strip() for x in str(f).split(',') if x.strip()]

tasks = read_tasks()
uid_map = {t['uuid']:t for t in tasks if 'uuid' in t}
succ=defaultdict(set); pred=defaultdict(set)
for t in tasks:
    u=t.get('uuid')
    if not u: continue
    for d in deps_list(t.get('depends')):
        if d in uid_map:
            succ[d].add(u); pred[u].add(d)

# build mermaid edges, collapse simple chains (simple heuristic)
visited=set(); edges=[]
for u in uid_map:
    if len(pred[u])!=1:
        for s in succ.get(u,[]):
            if (u,s) in visited: continue
            chain=[u]; cur=s
            while True:
                chain.append(cur); visited.add((chain[-2],cur))
                if len(pred[cur])==1 and len(succ[cur])==1:
                    cur = next(iter(succ[cur]))
                    if (chain[-1],cur) in visited: break
                else: break
            edges.append(chain)

# write mermaid to stdout (so you can redirect to graph.mmd)
print('flowchart TD')
for chain in edges:
    labels = [uid_map[x].get('description','').replace('\n',' ')[0:60].replace('"','') for x in chain]
    mer = ' --> '.join([f'"{labels[i]}\n({chain[i][:8]})"' for i in range(len(chain))])
    print('  '+mer)

# write CSV for Tana import (Title, Body, Tags, UUID, DependsOn)
with open('tasks.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(['Title','Body','Tags','UUID','DependsOn'])
    for u,t in uid_map.items():
        title = t.get('description','')
        body = f"task:{t.get('id','')} entry:{t.get('entry','') or ''}"
        tags = ';'.join(t.get('tags',[]))
        depends = ','.join(sorted(pred.get(u,[])))
        w.writerow([title,body,tags,u,depends])
