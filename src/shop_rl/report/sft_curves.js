(() => {
  const runs = JSON.parse(document.getElementById('sft-curve-data').textContent);
  const metric = document.getElementById('curve-metric');
  const axis = document.getElementById('curve-axis');
  const windowSize = document.getElementById('curve-window');
  const ns = 'http://www.w3.org/2000/svg';
  const specs = [
    {id:'train-loss', title:'训练总 loss', type:'train', key:'loss', note:'目标不同，比较各自趋势；不据绝对值排名。'},
    {id:'train-action', title:'训练行动段 CE', type:'train', key:'act_ce', note:'每步最后一个 micro-batch；格式 token 掩码略有差异。'},
    {id:'train-grad', title:'梯度范数（裁剪前）', type:'train', key:'grad_norm', reference:1, note:'虚线为裁剪阈值 1.0。'},
    {id:'val-loss', title:'验证总 loss', type:'validation', key:'val_loss', note:'40 条短样本；A/B 的监督目标不同。'},
    {id:'val-action', title:'验证行动段 CE', type:'validation', key:'val_act_ce', note:'各自行动掩码的验证均值。'},
    {id:'dev-success', title:'200 题开发集严格购买成功率', type:'evaluation', key:'success', reference:29, note:'同一批 200 题。虚线为原模型的 29.0%。'},
    {id:'train-thought', title:'A 的训练思考段 CE', type:'train', key:'think_ce', note:'该指标仅适用于有思考监督的 A。'}
  ];
  if (metric) {
    specs.forEach(spec => metric.add(new Option(spec.title, spec.id)));
    metric.value = 'train-action';
  }

  function element(tag, attrs = {}, text) {
    const e = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([k,v]) => e.setAttribute(k, v));
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function draw(spec, container) {
    const training = spec.type === 'train';
    const datasets = runs.map(run => {
      const rows = run[spec.type].filter(r => Number.isFinite(r[spec.key]));
      const width = training ? Number(windowSize.value) : 1;
      return {name:run.name, color:run.color, points:rows.map((r,i) => {
        const history = rows.slice(Math.max(0,i-width+1),i+1).filter(h => h.epoch === r.epoch);
        return {x:training ? r[axis.value] : r.epoch, raw:r[spec.key], step:r.step, epoch:r.epoch,
          y:history.reduce((sum,h) => sum+h[spec.key],0)/history.length};
      })};
    }).filter(d => d.points.length);
    const chart = document.createElement('div');
    chart.className = 'chart'; chart.dataset.chartId = spec.id;
    const title = document.createElement('h4'); title.textContent = spec.title; chart.append(title);
    const legend = document.createElement('div'); legend.className = 'lg';
    datasets.forEach(d => {
      const label = document.createElement('span'), sw = document.createElement('span');
      sw.className = 'sw'; sw.style.background = d.color;
      label.append(sw, document.createTextNode(d.name)); legend.append(label);
    });
    chart.append(legend);
    const W=metric ? 640 : 450, H=metric ? 250 : 230, left=48, right=18, top=15, bottom=38;
    const plotW=W-left-right, plotH=H-top-bottom;
    const xMax=training && axis.value==='step' ? Math.max(...datasets.flatMap(d=>d.points.map(p=>p.x))) : 3;
    const xMin=training ? 0 : 1;
    const allY=datasets.flatMap(d=>d.points.flatMap(p=>[p.y,p.raw]));
    const yMax=spec.type==='evaluation' ? 50 : Math.max(...allY, spec.reference || 0)*1.12;
    const sx=x=>left+(x-xMin)/(xMax-xMin)*plotW;
    const sy=y=>top+plotH*(1-y/yMax);
    const svg=element('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':spec.title+'：'+datasets.map(d=>d.name).join(' 与 ')});
    for(let i=0;i<=4;i++) {
      const y=yMax*i/4, py=sy(y);
      svg.append(element('line',{x1:left,y1:py,x2:W-right,y2:py,stroke:'#dfe4ea'}));
      const label = spec.type === 'evaluation' ? y.toFixed(0) + '%' : yMax < 1 ? y.toFixed(2) : y.toFixed(1);
      svg.append(element('text',{x:left-6,y:py+4,'text-anchor':'end',class:'ax'},label));
    }
    const xTicks=training && axis.value==='step' ? [0,Math.round(xMax/3),Math.round(2*xMax/3),xMax] : (training?[0,1,2,3]:[1,2,3]);
    xTicks.forEach(x=> {
      const px=sx(x);
      svg.append(element('line',{x1:px,y1:top,x2:px,y2:top+plotH,stroke:'#dfe4ea','stroke-dasharray':'3 5'}));
      svg.append(element('text',{x:px,y:top+plotH+15,'text-anchor':'middle',class:'ax'},String(x)));
    });
    svg.append(element('text',{x:left+plotW/2,y:H-3,'text-anchor':'middle',class:'ax'},training && axis.value==='step'?'优化步':'训练进度（epoch）'));
    if(spec.reference !== undefined) svg.append(element('line',{x1:left,y1:sy(spec.reference),x2:W-right,y2:sy(spec.reference),stroke:'#7b8490','stroke-dasharray':'6 4'}));
    datasets.forEach(d=> {
      const path=key=>d.points.map((p,i)=>(i?'L':'M')+sx(p.x).toFixed(2)+','+sy(p[key]).toFixed(2)).join(' ');
      if(training && Number(windowSize.value)>1) svg.append(element('path',{d:path('raw'),fill:'none',stroke:d.color,opacity:0.17,'stroke-width':1}));
      svg.append(element('path',{d:path('y'),fill:'none',stroke:d.color,'stroke-width':2.2,'data-series':d.name}));
      if(!training) d.points.forEach(p=>svg.append(element('circle',{cx:sx(p.x),cy:sy(p.y),r:3,fill:d.color})));
    });
    const cross=element('line',{y1:top,y2:top+plotH,stroke:'#7b8490','stroke-dasharray':'3 3',visibility:'hidden'});
    svg.append(cross);
    const dots=datasets.map(d=> {
      const dot=element('circle',{r:4,fill:d.color,stroke:'white','stroke-width':1.5,visibility:'hidden'});svg.append(dot);return dot;
    });
    const tip=document.createElement('div');tip.className='tip';
    chart.append(svg,tip);
    const foot=document.createElement('p');foot.className='curve-foot';foot.textContent=spec.note;chart.append(foot);
    svg.addEventListener('pointermove', e=> {
      const box=svg.getBoundingClientRect();
      const px=Math.min(W-right,Math.max(left,(e.clientX-box.left)/box.width*W));
      const x=xMin+(px-left)/plotW*(xMax-xMin);
      cross.setAttribute('x1',px);cross.setAttribute('x2',px);cross.setAttribute('visibility','visible');
      tip.replaceChildren();
      datasets.forEach((d,i)=> {
        const p=d.points.reduce((best,p)=>Math.abs(p.x-x)<Math.abs(best.x-x)?p:best);
        dots[i].setAttribute('cx',sx(p.x));dots[i].setAttribute('cy',sy(p.y));dots[i].setAttribute('visibility','visible');
        const line=document.createElement('div');
        const sw=document.createElement('span');sw.className='sw';sw.style.background=d.color;
        const value=document.createElement('b');value.textContent=p.y.toFixed(spec.type==='evaluation'?1:4)+(spec.type==='evaluation'?'%':'');
        line.append(sw,document.createTextNode(d.name+': '),value);
        const context=document.createElement('div');context.className='dim';
        context.textContent=training?`step ${p.step} · epoch ${p.epoch} · 原始 ${p.raw.toFixed(4)}`:`epoch ${p.epoch}`;
        tip.append(line,context);
      });
      tip.style.display='block';
      const localX=(e.clientX-chart.getBoundingClientRect().left)+10;
      tip.style.left=Math.max(6,Math.min(localX,chart.clientWidth-tip.offsetWidth-6))+'px';
      tip.style.top='42px';
    });
    svg.addEventListener('pointerleave',()=> {
      tip.style.display='none';cross.setAttribute('visibility','hidden');dots.forEach(d=>d.setAttribute('visibility','hidden'));
    });
    container.append(chart);
  }

  function render() {
    const main = document.getElementById('sft-main-charts');
    main.replaceChildren();
    if (metric) {
      const spec = specs.find(item => item.id === metric.value);
      axis.disabled = windowSize.disabled = spec.type !== 'train';
      draw(spec, main);
    } else {
      const thought = document.getElementById('sft-thought-chart');
      thought.replaceChildren();
      specs.slice(0, -1).forEach(spec => draw(spec, main));
      draw(specs.at(-1), thought);
    }
  }
  metric?.addEventListener('change',render);
  axis.addEventListener('change',render);
  windowSize.addEventListener('change',render);
  render();
})();
