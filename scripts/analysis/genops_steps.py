import gzip, json, glob, re, sys, collections, numpy as np
for run in sys.argv[1:]:
    R = {}
    for f in glob.glob('outputs/rt/gen_ops_grpo/%s/samples/step*.jsonl.gz' % run):
        R[int(re.search(r'step(\d+)', f).group(1))] = [json.loads(l) for l in gzip.open(f, 'rt', encoding='utf-8')]
    pc = lambda xs, f: 100 * np.mean([bool(f(x)) for x in xs]) if xs else float('nan')
    m = lambda xs, k: np.mean([(x.get(k) or 0.0) if x.get('why') == 'ok' else 0.0 for x in xs])
    print(run, 'steps', len(R)); print('window   mean p: real  nat   op  | all3% | r1    | critic>=.9  .1-.5 (of passing) | all3&critic<.5 | disc  | w     | low')
    for a in range(1, max(R) + 1, 10):
        xs = [x for k in range(a, a + 10) if k in R for x in R[k]]; g = [x for x in xs if x.get('gate')]
        low = collections.Counter(x.get('low', '') for x in xs if x.get('why') == 'ok').most_common(2)
        print('%3d-%3d        %.3f %.3f %.3f | %4.1f  | %.3f |   %3.0f%%      %3.0f%%            |    %4.1f%%      | %.3f | %.3f | %s' % (a, a + 9, m(xs, 'p_err'), m(xs, 'p_natural'), m(xs, 'p_op'), pc(xs, lambda x: x.get('gate')), m(xs, 'r1'), pc(g, lambda x: x['p_halluc'] >= .9), pc(g, lambda x: .1 <= x['p_halluc'] < .5), pc(xs, lambda x: x.get('gate') and x['p_halluc'] < .5), m(xs, 'p_disc'), np.mean([x.get('w_hard', 0) for x in xs if x.get('why') == 'ok']), low))
