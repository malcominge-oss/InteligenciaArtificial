"""Actividad 2.6: TSP empresarial simulado. Backend principal: DEAP.
El backend reference es independiente y no certifica la ejecución de DEAP.
"""
import argparse, csv, json, math, random, statistics, sys, platform
from pathlib import Path
from functools import lru_cache
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_data(path):
    with open(path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if [int(r['id']) for r in rows] != list(range(len(rows))):
        raise ValueError('Los ID deben ser consecutivos desde 0 (depósito).')
    if not 3 <= len(rows) <= 17:
        raise ValueError('Se admiten de 2 a 16 clientes para la verificación exacta.')
    points = [(float(r['x_km']), float(r['y_km'])) for r in rows]
    if not all(math.isfinite(v) for p in points for v in p):
        raise ValueError('Coordenadas no finitas.')
    return points


def matrix(points):
    return [[math.hypot(x-u,y-v) for u,v in points] for x,y in points]


def distance(ind, d):
    route = [0] + [x+1 for x in ind] + [0]
    return sum(d[a][b] for a,b in zip(route,route[1:]))


def valid(ind, n):
    return len(ind)==n and sorted(ind)==list(range(n))


def ordered(a,b):
    # Implementación de referencia de OX; no importa ni sustituye DEAP.
    n=len(a); lo,hi=sorted(random.sample(range(n),2))
    def child(p,q):
        out=[None]*n; out[lo:hi+1]=q[lo:hi+1]
        remaining=[p[(hi+1+i)%n] for i in range(n)
                   if p[(hi+1+i)%n] not in q[lo:hi+1]]
        slots=[(hi+1+i)%n for i in range(n) if not lo <= (hi+1+i)%n <= hi]
        for pos,val in zip(slots,remaining): out[pos]=val
        return out
    x,y=child(a,b),child(b,a); a[:]=x; b[:]=y
    return a,b


def shuffle(ind, indpb):
    for i in range(len(ind)):
        if random.random()<indpb:
            j=random.randint(0,len(ind)-2)
            if j>=i: j+=1
            ind[i],ind[j]=ind[j],ind[i]
    return ind,


def exact(d):
    n=len(d)-1
    @lru_cache(None)
    def cost(mask,j):
        previous=mask^(1<<j)
        if not previous: return d[0][j+1]
        return min(cost(previous,k)+d[k+1][j+1]
                   for k in range(n) if previous&(1<<k))
    mask=(1<<n)-1
    end=min(range(n),key=lambda j:cost(mask,j)+d[j+1][0])
    optimum=cost(mask,end)+d[end+1][0]
    rev=[]
    while mask:
        rev.append(end); previous=mask^(1<<end)
        if not previous: break
        end=min((k for k in range(n) if previous&(1<<k)),
                key=lambda k:cost(previous,k)+d[k+1][end+1])
        mask=previous
    route=rev[::-1]
    assert valid(route,n) and math.isclose(distance(route,d),optimum,rel_tol=1e-12)
    return optimum,route


def nearest(d):
    remaining=set(range(1,len(d))); route=[]; last=0
    while remaining:
        nxt=min(remaining,key=lambda j:(d[last][j],j))
        remaining.remove(nxt); route.append(nxt-1); last=nxt
    return route


def run_ga(d, seed, backend='deap', popsize=200, generations=30,
           cxpb=.85, mutpb=.25, indpb=.08, elite=2):
    random.seed(seed); n=len(d)-1
    if backend=='deap':
        from deap import base, creator, tools
        if not hasattr(creator,'FitnessMin26'):
            creator.create('FitnessMin26',base.Fitness,weights=(-1.0,))
            creator.create('Route26',list,fitness=creator.FitnessMin26)
        tb=base.Toolbox()
        tb.register('indices',random.sample,range(n),n)
        tb.register('individual',tools.initIterate,creator.Route26,tb.indices)
        tb.register('population',tools.initRepeat,list,tb.individual)
        tb.register('evaluate',lambda x:(distance(x,d),))
        tb.register('select',tools.selTournament,tournsize=3)
        tb.register('mate',tools.cxOrdered)
        tb.register('mutate',tools.mutShuffleIndexes,indpb=indpb)
        pop=tb.population(n=popsize)
        for x in pop: x.fitness.values=tb.evaluate(x)
        mate,mutate=tb.mate,tb.mutate
    else:
        pop=[random.sample(range(n),n) for _ in range(popsize)]
        mate=ordered; mutate=lambda x:shuffle(x,indpb)
    history=[]
    for gen in range(generations+1):
        values=[distance(x,d) for x in pop]
        history.append(dict(seed=seed,generation=gen,best_km=min(values),
                            mean_km=statistics.mean(values),
                            diversity=len({tuple(x) for x in pop})/popsize))
        if gen==generations: break
        elites=[deepcopy(x) for x in sorted(pop,key=lambda x:distance(x,d))[:elite]]
        if backend=='deap':
            children=[tb.clone(x) for x in tb.select(pop,popsize-elite)]
        else:
            children=[deepcopy(min([random.choice(pop) for _ in range(3)],
                      key=lambda x:distance(x,d))) for _ in range(popsize-elite)]
        for a,b in zip(children[::2],children[1::2]):
            if random.random()<cxpb: mate(a,b)
        for x in children:
            if random.random()<mutpb: mutate(x)
            assert valid(x,n), 'Permutación inválida'
            if backend=='deap':
                del x.fitness.values
                x.fitness.values=tb.evaluate(x)
        pop=elites+children
    assert all(history[i+1]['best_km']<=history[i]['best_km']+1e-10
               for i in range(generations))
    assert len(history)==31 and len(pop)==popsize
    best=min(pop,key=lambda x:distance(x,d))
    return list(best),history


def write_csv(path, rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['deap','reference'],default='deap')
    parser.add_argument('--data',default=str(Path(__file__).with_name('clientes.csv')))
    parser.add_argument('--out',default='resultados_deap')
    args=parser.parse_args()
    if args.backend=='deap':
        try: import deap
        except ImportError:
            raise SystemExit('Falta DEAP. Instala: python -m pip install -r requirements.txt')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    points=read_data(args.data); d=matrix(points); n=len(points)-1
    assert all(abs(d[i][j]-d[j][i])<1e-12 for i in range(n+1) for j in range(n+1))
    optimum,opt_route=exact(d); nn=nearest(d); original=list(range(n))
    histories=[]; runs=[]; routes=[]
    for seed in range(30):
        route,h=run_ga(d,seed,args.backend); length=distance(route,d)
        histories+=h; routes.append(route)
        # Comparador aleatorio con igual cantidad de candidatos puntuados.
        rng=random.Random(10000+seed)
        budget=200+30*198
        random_best=min(distance(rng.sample(range(n),n),d) for _ in range(budget))
        runs.append(dict(seed=seed,best_km=length,gap_pct=100*(length/optimum-1),
                         random_best_km=random_best))
    chosen=min(range(30),key=lambda i:runs[i]['best_km'])
    best=routes[chosen]; lengths=[r['best_km'] for r in runs]
    meta=dict(backend=args.backend,python=platform.python_version(),
              deap_version=getattr(sys.modules.get('deap'),'__version__',None),
              points=len(points),customers=n,population=200,generations=30,
              runs=30,seeds=list(range(30)),cxpb=.85,mutpb=.25,indpb=.08,
              tournament=3,elite=2,best_seed=chosen,best_km=min(lengths),
              mean_km=statistics.mean(lengths),std_km=statistics.stdev(lengths),
              worst_km=max(lengths),exact_km=optimum,
              hits=sum(math.isclose(x,optimum,abs_tol=1e-8) for x in lengths),
              original_km=distance(original,d),nearest_km=distance(nn,d),
              random_mean_km=statistics.mean(r['random_best_km'] for r in runs),
              route=[0]+[i+1 for i in best]+[0],
              exact_route=[0]+[i+1 for i in opt_route]+[0],
              validation='Permutaciones, matriz, elitismo, 31 registros y costo exacto: OK')
    (out/'resumen.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    write_csv(out/'corridas.csv',runs); write_csv(out/'historial.csv',histories)
    full=[0]+[x+1 for x in best]+[0]; acc=0; legs=[]
    for k,(a,b) in enumerate(zip(full,full[1:]),1):
        acc+=d[a][b]; legs.append(dict(tramo=k,origen=a,destino=b,km=d[a][b],acumulado_km=acc))
    write_csv(out/'ruta.csv',legs)
    write_csv(out/'distancias.csv',[dict(origen=i,**{f'nodo_{j}':d[i][j] for j in range(n+1)}) for i in range(n+1)])
    plt.figure(figsize=(8,4.5))
    selected=[r for r in histories if r['seed']==chosen]
    plt.plot(range(31),[r['best_km'] for r in selected],label=f'Mejor, semilla {chosen}')
    plt.plot(range(31),[r['mean_km'] for r in selected],label='Media poblacional')
    plt.axhline(optimum,color='black',ls='--',label='Óptimo exacto')
    plt.xlabel('Generación (0 = población inicial)'); plt.ylabel('Distancia (km)')
    plt.title(f'Convergencia — backend {args.backend}'); plt.legend(); plt.tight_layout()
    plt.savefig(out/'convergencia.png',dpi=170); plt.close()
    plt.figure(figsize=(8,4.5))
    for seed in range(30):
        plt.plot(range(31),[r['best_km'] for r in histories if r['seed']==seed],alpha=.18,color='steelblue')
    plt.plot(range(31),[statistics.mean(r['best_km'] for r in histories if r['generation']==g) for g in range(31)],color='navy',label='Promedio de mejores (30 corridas)')
    plt.axhline(optimum,color='black',ls='--',label='Óptimo exacto')
    plt.xlabel('Generación'); plt.ylabel('Distancia (km)'); plt.legend(); plt.tight_layout()
    plt.savefig(out/'convergencia_30.png',dpi=170); plt.close()
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,route,title in [(axes[0],[0]+list(range(1,n+1))+[0],'Orden de captura'),(axes[1],full,'Mejor ruta AG')]:
        ax.plot([points[i][0] for i in route],[points[i][1] for i in route],'-o',ms=4)
        for i,(x,y) in enumerate(points): ax.annotate(str(i),(x,y),xytext=(4,4),textcoords='offset points')
        ax.scatter(*points[0],marker='s',s=80,color='red',zorder=5)
        ax.set(title=title,xlabel='x (km)',ylabel='y (km)'); ax.set_aspect('equal'); ax.grid(alpha=.2)
    fig.suptitle(f'Caso simulado — backend {args.backend}'); fig.tight_layout()
    fig.savefig(out/'rutas.png',dpi=170); plt.close(fig)
    print(json.dumps(meta,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
