from pathlib import Path
import argparse,json,sys
import video_to_intervals as annotation

EXTENSIONS={'.mp4','.qt','.mov','.mkv','.avi','.m4v'}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',nargs='+',type=Path,help='Video paths or directories')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--view',choices=['auto','front','side'],default='auto')
    p.add_argument('--task',choices=['squats','activities'],default='squats')
    p.add_argument('--action-config',type=Path)
    p.add_argument('--font',type=Path)
    p.add_argument('--layout',choices=['single','front-side','side-front'],default='single',
                   help='Panel order; dual panels must show the same synchronized performance')
    p.add_argument('--layout-map',type=Path,help='JSON mapping video stem or filename to layout, optionally split_ratio')
    p.add_argument('--split-ratio',type=float,help='Left-panel width / full display width (default 0.5)')
    p.add_argument('--fusion-config',type=Path)
    p.add_argument('--resume',action='store_true',help='Reuse dual-view pose caches only after source/model/crop checks')
    p.add_argument('--device',default='')
    p.add_argument('--track-id',type=int)
    p.add_argument('--no-preview',action='store_true')
    p.add_argument('--labels-only',action='store_true')
    p.add_argument('--grading-config',type=Path)
    p.add_argument('--recursive',action='store_true')
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parent/'final_parameters.json')
    args,extra=p.parse_known_args(argv)
    config=json.loads(args.config.read_text(encoding='utf-8'))
    configured=config['parameters']
    defaults=vars(annotation.parser().parse_args(['--output','unused']))
    forbidden={'video','output','keypoints','preview','model','device','track_id'}
    if set(configured)&forbidden or set(configured)-set(defaults):p.error('Unsupported detection parameter in config')
    fixed=[]
    for name,value in configured.items():
        if value is not None:fixed.extend(['--'+name.replace('_','-'),str(value)])
    model=Path(config['model'])
    if not model.is_absolute():model=args.config.resolve().parent/model
    if not model.is_file():p.error('Model file missing: '+str(model))
    fixed+=['--model',str(model)]
    explicit_view=any(x=='--view' or x.startswith('--view=') for x in (sys.argv[1:] if argv is None else argv))
    sources=[]
    for path in args.inputs:
        if path.is_dir():
            sources.extend(sorted(x for x in (path.rglob('*') if args.recursive else path.iterdir()) if x.is_file() and x.suffix.lower() in EXTENSIONS))
        elif path.is_file() and path.suffix.lower() in EXTENSIONS:sources.append(path)
        else:p.error('Missing or unsupported input: '+str(path))
    sources=list(dict.fromkeys(x.resolve() for x in sources))
    if not sources:p.error('No videos found')
    stems=[x.stem.casefold() for x in sources]
    if len(stems)!=len(set(stems)):p.error('Input stems collide; use separate output directories')
    args.output.mkdir(parents=True,exist_ok=True)
    failures=[]
    layouts=json.loads(args.layout_map.read_text(encoding='utf-8-sig')) if args.layout_map else {}
    if not isinstance(layouts,dict):p.error('Layout map must be a JSON object')
    for i,source in enumerate(sources,1):
        dest=args.output/source.stem
        print('VIDEO_START {}/{} {}'.format(i,len(sources),source),flush=True)
        cmd=[str(source),'--output',str(dest)]+fixed+['--device',args.device or config.get('device','0')]
        if explicit_view:cmd+=['--view',args.view]
        if args.track_id is not None:cmd+=['--track-id',str(args.track_id)]
        if args.labels_only and not args.no_preview:cmd+=['--preview']
        try:
            selected=layouts.get(source.name,layouts.get(source.stem,args.layout))
            selected=selected if isinstance(selected,dict) else {'layout':selected}
            layout=selected.get('layout',args.layout)
            if layout not in ('single','front-side','side-front'):raise ValueError('Unsupported layout: '+str(layout))
            if args.task=='activities':
                if layout!='single':raise ValueError('Activity mode currently requires one full camera image; use --task squats for synchronized dual panels.')
                from action_pipeline import run as action_run
                action_run(source,dest,cmd+extra,config_path=args.action_config,no_preview=args.no_preview,resume=args.resume,font=args.font)
                continue
            if layout!='single':
                from multiview_pipeline import run as dual_run
                dual_run(source,dest,cmd+extra,layout=layout,split_ratio=selected.get('split_ratio',args.split_ratio),
                         fusion_config=args.fusion_config,grading_config=args.grading_config,
                         no_preview=args.no_preview,labels_only=args.labels_only,resume=args.resume)
                continue
            annotation.main(cmd+extra)
            if not args.labels_only:
                from grade_repetitions import run as grade, render_graded_video
                _,grades=grade(dest,None,dest/'grading',mapped=False,config_path=args.grading_config)
                if not args.no_preview:render_graded_video(source,dest,grades)
            print('VIDEO_READY '+str(dest/'review.mp4' if not args.no_preview else dest/'intervals.csv'),flush=True)
        except Exception as error:
            failures.append(dict(source=str(source),error=repr(error)))
            print('VIDEO_FAILED '+str(source)+' '+repr(error),file=sys.stderr,flush=True)
    (args.output/'batch_report.json').write_text(json.dumps(dict(inputs=[str(x) for x in sources],configuration=str(args.config.resolve()),
         grading='disabled_for_activity_mode' if args.task=='activities' else 'disabled' if args.labels_only else 'v4.5 dual-view fusion / legacy single-view observable video rules',task=args.task,
         default_layout=args.layout,layout_map=str(args.layout_map) if args.layout_map else None,failures=failures),ensure_ascii=False,indent=2),encoding='utf-8')
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
