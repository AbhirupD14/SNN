import numpy as np
import json
from fep_model import FEPSNN

def generate_html_viz(traces):
    # Convert numpy types to native python types for JSON serialization
    def sanitize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(i) for i in obj]
        return obj

    json_data = json.dumps(sanitize(traces))
    
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FEP SNN | Neuromorphic Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-deep: #0a0a0c;
            --bg-panel: rgba(22, 22, 26, 0.8);
            --border-color: #2d3139;
            --text-main: #e6edf3;
            --text-dim: #8b949e;
            --accent-cyan: #00f2ff;
            --accent-emerald: #00ffaa;
            --accent-amber: #ffb800;
            --accent-red: #ff4d4d;
            --glass-bg: rgba(25, 25, 30, 0.75);
        }}

        body {{ 
            background: var(--bg-deep); 
            color: var(--text-main); 
            font-family: 'Inter', sans-serif; 
            margin: 0; 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            min-height: 100vh;
            overflow-x: hidden;
        }}

        #header {{ 
            width: 100%; 
            padding: 24px 0; 
            text-align: center; 
            background: linear-gradient(to bottom, #161b22, transparent);
            border-bottom: 1px solid var(--border-color);
        }}

        h1 {{ 
            font-weight: 600; 
            letter-spacing: -0.02em; 
            margin: 0; 
            font-size: 24px; 
            color: var(--text-main);
        }}

        .subtitle {{ 
            color: var(--text-dim); 
            font-size: 13px; 
            font-family: 'JetBrains Mono', monospace;
            margin-top: 8px;
        }}

        #dashboard {{ 
            display: grid; 
            grid-template-columns: 240px 600px 320px; 
            gap: 24px; 
            width: 1160px; 
            margin-top: 40px; 
            align-items: start;
        }}

        .panel {{ 
            background: var(--bg-panel); 
            border: 1px solid var(--border-color); 
            border-radius: 12px; 
            padding: 20px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            backdrop-filter: blur(8px);
        }}

        .panel-title {{ 
            font-size: 12px; 
            text-transform: uppercase; 
            letter-spacing: 0.1em; 
            color: var(--text-dim); 
            margin-bottom: 20px; 
            font-weight: 700;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
        }}

        #canvas-container {{ 
            position: relative; 
            width: 600px; 
            height: 600px; 
            background: radial-gradient(circle, #111 0%, #000 100%); 
            border: 1px solid var(--border-color); 
            border-radius: 16px; 
            overflow: hidden; 
            box-shadow: inset 0 0 50px rgba(0,0,0,1);
        }}

        canvas {{ position: absolute; top: 0; left: 0; }}

        #metrics-container {{ 
            display: flex; 
            flex-direction: column; 
            gap: 10px; 
        }}

        .metric-card {{ 
            background: var(--glass-bg); 
            border: 1px solid var(--border-color); 
            border-radius: 8px; 
            padding: 12px; 
            display: flex; 
            justify-content: space-between; 
            align-items: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            transition: border-color 0.2s;
        }}

        .metric-card:hover {{ border-color: var(--accent-cyan); }}

        .metric-label {{ color: var(--text-dim); }}
        .metric-val {{ color: var(--accent-cyan); font-weight: 700; }}

        #controls-bar {{ 
            margin-top: 40px; 
            width: 1160px; 
            background: var(--bg-panel); 
            border: 1px solid var(--border-color); 
            border-radius: 16px; 
            padding: 20px; 
            display: flex; 
            flex-direction: column; 
            gap: 20px;
            box-sizing: border-box;
        }}

        .btn-group {{ 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            gap: 16px; 
        }}

        button {{ 
            background: #21262d; 
            color: var(--text-main); 
            border: 1px solid var(--border-color); 
            padding: 8px 20px; 
            cursor: pointer; 
            border-radius: 6px; 
            font-family: 'Inter', sans-serif; 
            font-weight: 600; 
            font-size: 14px; 
            transition: all 0.2s ease;
        }}

        button:hover {{ 
            background: #30363d; 
            border-color: #8b949e; 
            transform: translateY(-1px);
        }}

        button:active {{ transform: translateY(0); }}

        #playBtn.active {{ 
            background: var(--accent-cyan); 
            color: #000; 
            border-color: var(--accent-cyan); 
        }}

        #step-info {{ 
            font-family: 'JetBrains Mono', monospace; 
            font-size: 13px; 
            color: var(--text-dim); 
            min-width: 120px; 
            text-align: center;
        }}

        #timeline-container {{ 
            position: relative; 
            width: 100%; 
        }}

        input[type=range] {{ 
            width: 100%; 
            appearance: none; 
            background: #30363d; 
            height: 4px; 
            border-radius: 2px; 
            outline: none;
        }}

        input[type=range]::-webkit-slider-thumb {{ 
            appearance: none; 
            width: 16px; 
            height: 16px; 
            background: var(--accent-cyan); 
            border-radius: 50%; 
            cursor: pointer; 
            box-shadow: 0 0 10px var(--accent-cyan);
        }}

        .status-banner {{ 
            font-family: 'JetBrains Mono', monospace; 
            font-size: 14px; 
            text-align: center; 
            color: var(--accent-emerald); 
            margin-bottom: 20px; 
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>FEP SNN : NEUROMORPHIC RACE</h1>
        <div class="subtitle">Computational Neuroscience | Local Variational Free Energy</div>
    </div>

    <div id="dashboard">
        <div class="panel">
            <div class="panel-title">Input Vector</div>
            <div style="display: flex; justify-content: center;">
                <canvas id="inputCanvas" width="150" height="150"></canvas>
            </div>
        </div>

        <div id="canvas-container">
            <canvas id="snnCanvas" width="600" height="600"></canvas>
        </div>

        <div class="panel">
            <div class="panel-title">Live Telemetry</div>
            <div id="metrics-container"></div>
        </div>
    </div>

    <div id="controls-bar">
        <div class="btn-group">
            <button onclick="prevStep()">« Previous</button>
            <button onclick="togglePlay()" id="playBtn">Play</button>
            <button onclick="nextStep()">Next »</button>
            <div id="step-info">Step: 0 / 0</div>
        </div>
        <div id="timeline-container">
            <input type="range" id="scrubber" min="0" max="0" value="0" oninput="goToStep(this.value)">
        </div>
    </div>

    <script>
        const data = {json_data};
        let currentStep = 0;
        let isPlaying = false;
        let playInterval;

        const inputCanvas = document.getElementById('inputCanvas');
        const inputCtx = inputCanvas.getContext('2d');
        const snnCanvas = document.getElementById('snnCanvas');
        const snnCtx = snnCanvas.getContext('2d');
        const metricsDiv = document.getElementById('metrics-container');
        const stepInfo = document.getElementById('step-info');
        const scrubber = document.getElementById('scrubber');

        scrubber.max = data.length - 1;

        function draw() {{
            const state = data[currentStep];
            const {{ v_e, theta_e, pattern, stage, extra }} = state;

            // Draw Input Vector
            inputCtx.clearRect(0, 0, 150, 150);
            const cellSize = 46;
            for(let i=0; i<9; i++) {{
                const x = (i % 3) * cellSize;
                const y = Math.floor(i / 3) * cellSize;
                inputCtx.fillStyle = pattern[i] > 0 ? '#00f2ff' : '#1a1a22';
                inputCtx.strokeStyle = '#2d3139';
                inputCtx.lineWidth = 1;
                inputCtx.fillRect(x+2, y+2, cellSize-4, cellSize-4);
                inputCtx.strokeRect(x+2, y+2, cellSize-4, cellSize-4);
                if(pattern[i] > 0) {{
                    inputCtx.shadowBlur = 10;
                    inputCtx.shadowColor = '#00f2ff';
                    inputCtx.strokeRect(x+6, y+6, cellSize-12, cellSize-12);
                    inputCtx.shadowBlur = 0;
                }}
            }}

            // Draw SNN
            snnCtx.clearRect(0, 0, 600, 600);
            
            // I_FB boundary
            snnCtx.strokeStyle = '#2d3139';
            snnCtx.lineWidth = 2;
            snnCtx.strokeRect(100, 100, 400, 400);

            // E-neurons
            const centerX = 300, centerY = 300;
            const spacing = 140;
            for(let i=0; i<9; i++) {{
                const x = centerX + ((i % 3) - 1) * spacing;
                const y = centerY + (Math.floor(i / 3) - 1) * spacing;
                
                const volt = v_e[i];
                const thresh = theta_e[i];
                const ratio = Math.min(volt / thresh, 1.5);
                
                if (stage === 'e_spike' && extra.winner === i) {{
                    snnCtx.beginPath();
                    snnCtx.arc(x, y, 35, 0, Math.PI*2);
                    snnCtx.fillStyle = 'rgba(0, 255, 170, 0.3)';
                    snnCtx.fill();
                    snnCtx.strokeStyle = '#00ffaa';
                    snnCtx.lineWidth = 4;
                    snnCtx.stroke();
                }}

                snnCtx.beginPath();
                snnCtx.arc(x, y, 22, 0, Math.PI*2);
                snnCtx.fillStyle = `rgb(${{0}}, ${{Math.floor(40 + ratio * 100)}}, ${{Math.floor(80 + ratio * 150)}})`;
                snnCtx.fill();
                
                snnCtx.strokeStyle = volt >= thresh ? '#fff' : '#2d3139';
                snnCtx.lineWidth = 3;
                snnCtx.stroke();
                
                snnCtx.fillStyle = '#8b949e';
                snnCtx.font = '12px JetBrains Mono';
                snnCtx.fillText(`E${{i}}`, x-10, y-30);
            }}

            // I_FF blocks
            const iffPos = [[150, 150], [450, 150], [150, 450], [450, 450]];
            iffPos.forEach((pos, i) => {{
                snnCtx.fillStyle = (stage === 'iff_shunt') ? '#ff4d4d' : '#1a1a22';
                snnCtx.strokeStyle = '#ff4d4d';
                snnCtx.lineWidth = 2;
                snnCtx.fillRect(pos[0]-20, pos[1]-20, 40, 40);
                snnCtx.strokeRect(pos[0]-20, pos[1]-20, 40, 40);
                snnCtx.fillStyle = '#8b949e';
                snnCtx.fillText(`Iff${{i}}`, pos[0]-15, pos[1]-30);
            }});

            if(stage === 'iff_shunt') {{
                snnCtx.strokeStyle = 'rgba(255, 77, 77, 0.4)';
                snnCtx.lineWidth = 2;
                iffPos.forEach(pos => {{
                    snnCtx.beginPath();
                    snnCtx.moveTo(pos[0], pos[1]);
                    snnCtx.lineTo(centerX, centerY);
                    snnCtx.stroke();
                }});
            }}

            // Metrics
            let mHtml = `<div class="status-banner">${{stage.replace('_', ' ')}}</div>`;
            for(let i=0; i<9; i++) {{
                const freeEnergy = Math.max(0, v_e[i] - theta_e[i]);
                mHtml += `<div class="metric-card">
                    <span class="metric-label">Neuron ${{i}}</span>
                    <span class="metric-val">V:${{v_e[i].toFixed(1)}} | T:${{theta_e[i].toFixed(1)}} | F:${{freeEnergy.toFixed(1)}}</span>
                </div>`;
            }}
            metricsDiv.innerHTML = mHtml;
            stepInfo.innerText = `Step: ${{currentStep}} / ${{data.length - 1}}`;
            scrubber.value = currentStep;
        }}

        function nextStep() {{
            if(currentStep < data.length - 1) {{
                currentStep++;
                draw();
            }} else {{
                isPlaying = false;
                clearInterval(playInterval);
                document.getElementById('playBtn').innerText = 'Play';
                document.getElementById('playBtn').classList.remove('active');
            }}
        }}

        function prevStep() {{
            if(currentStep > 0) {{
                currentStep--;
                draw();
            }}
        }}

        function goToStep(val) {{
            currentStep = parseInt(val);
            draw();
        }}

        function togglePlay() {{
            isPlaying = !isPlaying;
            const btn = document.getElementById('playBtn');
            btn.innerText = isPlaying ? 'Pause' : 'Play';
            if(isPlaying) btn.classList.add('active'); else btn.classList.remove('active');
            if(isPlaying) {{
                playInterval = setInterval(nextStep, 500);
            }} else {{
                clearInterval(playInterval);
            }}
        }}

        draw();
    </script>
</body>
</html>
    """
    return html_template

def main():
    # For the purpose of updating the style, we can re-run the a single race 
    # or use existing data. We'll run a fresh race to generate a clean trace.
    snn = FEPSNN()
    pattern = np.ones(9)
    traces = snn.process_event_pattern(pattern)
    
    html_content = generate_html_viz(traces)
    with open("/home/adasgup/projects/sim_snn_fep/index.html", "w") as f:
        f.write(html_content)
    print("Stylized index.html written to /home/adasgup/projects/sim_snn_fep/index.html")

if __name__ == "__main__":
    main()
