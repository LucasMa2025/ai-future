"""
AGA Experiment Tool - Web Interface

A simple web tool for AGA experiments:
- Knowledge injection with persistence
- Multi-model support (DeepSeek, Qwen, Ollama, vLLM)
- Experiment data collection
- Statistics and monitoring

Usage:
    python app.py [--config config.yaml] [--port 8765]
"""
import os
import sys
import json
import yaml
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for
from functools import wraps

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== Configuration ====================

@dataclass
class Config:
    """Application configuration"""
    auth_enabled: bool = True
    auth_password: str = "aga_experiment_2026"
    host: str = "0.0.0.0"
    port: int = 8765
    debug: bool = False
    default_model: str = "gpt2"
    aga_bottleneck_dim: int = 64
    aga_num_slots: int = 100
    aga_target_layers: List[int] = None
    output_dir: str = "./experiment_results"
    db_path: str = "./aga_data.db"
    
    def __post_init__(self):
        if self.aga_target_layers is None:
            self.aga_target_layers = [-4, -3, -2, -1]
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls(
            auth_enabled=data.get('auth', {}).get('enabled', True),
            auth_password=data.get('auth', {}).get('password', 'aga_experiment_2026'),
            host=data.get('server', {}).get('host', '0.0.0.0'),
            port=data.get('server', {}).get('port', 8765),
            debug=data.get('server', {}).get('debug', False),
            default_model=data.get('models', {}).get('default', 'gpt2'),
            aga_bottleneck_dim=data.get('aga', {}).get('bottleneck_dim', 64),
            aga_num_slots=data.get('aga', {}).get('num_slots', 100),
            aga_target_layers=data.get('aga', {}).get('target_layers', [-4, -3, -2, -1]),
            output_dir=data.get('experiment', {}).get('output_dir', './experiment_results'),
            db_path=data.get('persistence', {}).get('db_path', './aga_data.db'),
        )


# ==================== AGA Manager ====================

class AGAExperimentManager:
    """Manages AGA experiments across multiple models with persistence"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.aga_modules: Dict[str, Any] = {}
        self.persistence_managers: Dict[str, Any] = {}
        self.experiment_results: List[Dict] = []
        
        # Ensure output directory exists
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize persistence
        from aga.persistence import SQLitePersistence
        self.persistence = SQLitePersistence(config.db_path)
        logger.info(f"Persistence initialized: {config.db_path}")
    
    def load_model(self, model_name: str) -> bool:
        """Load a model and create AGA module"""
        if model_name in self.models:
            return True
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Map model names to HuggingFace IDs
            model_mapping = {
                'gpt2': 'gpt2',
                'gpt2-medium': 'gpt2-medium',
                'gpt2-large': 'gpt2-large',
                'llama-2-7b': 'meta-llama/Llama-2-7b-hf',
                'llama-3.2-1b': 'meta-llama/Llama-3.2-1B',
                'mistral-7b': 'mistralai/Mistral-7B-v0.1',
                'qwen-1.8b': 'Qwen/Qwen-1_8B',
                'qwen2-0.5b': 'Qwen/Qwen2-0.5B',
                'qwen2.5-0.5b': 'Qwen/Qwen2.5-0.5B',
                'deepseek-coder-1.3b': 'deepseek-ai/deepseek-coder-1.3b-instruct',
            }
            
            model_id = model_mapping.get(model_name, model_name)
            
            logger.info(f"Loading model: {model_id}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            model.eval()
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            # Create AGA module
            self._create_aga_module(model_name)
            
            # Load persisted knowledge
            self._load_persisted_knowledge(model_name)
            
            logger.info(f"Model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_aga_module(self, model_name: str):
        """Create AGA module for a model"""
        from aga.core import AuxiliaryGovernedAttention, AGAManager
        from aga.persistence import AGAPersistenceManager
        
        model = self.models[model_name]
        
        # Get model dimensions
        if hasattr(model.config, 'hidden_size'):
            hidden_dim = model.config.hidden_size
        elif hasattr(model.config, 'n_embd'):
            hidden_dim = model.config.n_embd
        else:
            hidden_dim = 768
        
        if hasattr(model.config, 'num_attention_heads'):
            num_heads = model.config.num_attention_heads
        elif hasattr(model.config, 'n_head'):
            num_heads = model.config.n_head
        else:
            num_heads = 12
        
        # Create standalone AGA (simpler for demo)
        aga = AuxiliaryGovernedAttention(
            hidden_dim=hidden_dim,
            bottleneck_dim=self.config.aga_bottleneck_dim,
            num_slots=self.config.aga_num_slots,
            num_heads=num_heads,
        )
        aga.eval()
        
        if torch.cuda.is_available():
            aga = aga.cuda()
        
        self.aga_modules[model_name] = aga
        
        # Create persistence manager
        pm = AGAPersistenceManager(self.persistence, aga_id=model_name)
        self.persistence_managers[model_name] = pm
        
        logger.info(f"AGA created for {model_name}: hidden_dim={hidden_dim}, num_heads={num_heads}")
    
    def _load_persisted_knowledge(self, model_name: str):
        """Load persisted knowledge into AGA"""
        pm = self.persistence_managers.get(model_name)
        aga = self.aga_modules.get(model_name)
        
        if pm and aga:
            try:
                pm.load_aga(aga)
                logger.info(f"Loaded persisted knowledge for {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load persisted knowledge: {e}")
    
    def inject_knowledge(
        self,
        model_name: str,
        condition: str,
        decision: str,
        lifecycle_state: str = 'probationary',
    ) -> Dict[str, Any]:
        """Inject knowledge into AGA with persistence"""
        if model_name not in self.models:
            return {"success": False, "error": f"Model {model_name} not loaded"}
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        aga = self.aga_modules.get(model_name)
        pm = self.persistence_managers.get(model_name)
        
        if not aga:
            return {"success": False, "error": "AGA not initialized"}
        
        try:
            from aga.core import LifecycleState
            
            # Encode condition and decision
            with torch.no_grad():
                device = next(model.parameters()).device
                
                condition_tokens = tokenizer(condition, return_tensors="pt").to(device)
                decision_tokens = tokenizer(decision, return_tensors="pt").to(device)
                
                # Get embeddings
                if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                    embed_layer = model.model.embed_tokens
                elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
                    embed_layer = model.transformer.wte
                elif hasattr(model, 'get_input_embeddings'):
                    embed_layer = model.get_input_embeddings()
                else:
                    return {"success": False, "error": "Cannot find embedding layer"}
                
                condition_emb = embed_layer(condition_tokens.input_ids).mean(dim=1)
                decision_emb = embed_layer(decision_tokens.input_ids).mean(dim=1)
                
                key_vector = condition_emb[0]
                value_vector = decision_emb[0]
            
            # Generate LU ID
            lu_id = f"LU_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(condition)%10000:04d}"
            
            # Map lifecycle state
            state_map = {
                'probationary': LifecycleState.PROBATIONARY,
                'confirmed': LifecycleState.CONFIRMED,
                'deprecated': LifecycleState.DEPRECATED,
                'quarantined': LifecycleState.QUARANTINED,
            }
            lc_state = state_map.get(lifecycle_state, LifecycleState.PROBATIONARY)
            
            # Inject with persistence
            if pm:
                slot_idx = pm.sync_knowledge(
                    aga=aga,
                    lu_id=lu_id,
                    condition=condition,
                    decision=decision,
                    key_vector=key_vector,
                    value_vector=value_vector,
                    lifecycle_state=lc_state,
                )
            else:
                slot_idx = aga.find_free_slot()
                if slot_idx is not None:
                    aga.inject_knowledge(
                        slot_idx=slot_idx,
                        key_vector=key_vector,
                        value_vector=value_vector,
                        lu_id=lu_id,
                        lifecycle_state=lc_state,
                        condition=condition,
                        decision=decision,
                    )
            
            if slot_idx is None:
                return {"success": False, "error": "No free slots available"}
            
            result = {
                "success": True,
                "lu_id": lu_id,
                "slot_idx": slot_idx,
                "condition": condition,
                "decision": decision,
                "lifecycle_state": lifecycle_state,
                "timestamp": datetime.now().isoformat(),
            }
            
            logger.info(f"Knowledge injected: {lu_id} at slot {slot_idx}")
            return result
            
        except Exception as e:
            logger.error(f"Injection failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def update_lifecycle(
        self,
        model_name: str,
        lu_id: str,
        new_state: str,
    ) -> Dict[str, Any]:
        """Update knowledge lifecycle state"""
        aga = self.aga_modules.get(model_name)
        pm = self.persistence_managers.get(model_name)
        
        if not aga:
            return {"success": False, "error": "AGA not initialized"}
        
        try:
            from aga.core import LifecycleState
            
            state_map = {
                'probationary': LifecycleState.PROBATIONARY,
                'confirmed': LifecycleState.CONFIRMED,
                'deprecated': LifecycleState.DEPRECATED,
                'quarantined': LifecycleState.QUARANTINED,
            }
            lc_state = state_map.get(new_state, LifecycleState.PROBATIONARY)
            
            if pm:
                pm.sync_lifecycle_update(aga, lu_id, lc_state)
            else:
                slots = aga.get_slot_by_lu_id(lu_id)
                for slot_idx in slots:
                    aga.update_lifecycle(slot_idx, lc_state)
            
            return {"success": True, "lu_id": lu_id, "new_state": new_state}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def quarantine_knowledge(self, model_name: str, lu_id: str) -> Dict[str, Any]:
        """Quarantine knowledge"""
        aga = self.aga_modules.get(model_name)
        pm = self.persistence_managers.get(model_name)
        
        if not aga:
            return {"success": False, "error": "AGA not initialized"}
        
        try:
            if pm:
                pm.sync_quarantine(aga, lu_id)
            else:
                aga.quarantine_by_lu_id(lu_id)
            
            return {"success": True, "lu_id": lu_id}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run_inference(
        self,
        model_name: str,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> Dict[str, Any]:
        """Run inference with AGA"""
        if model_name not in self.models:
            return {"success": False, "error": f"Model {model_name} not loaded"}
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        try:
            device = next(model.parameters()).device
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "success": True,
                "prompt": prompt,
                "response": response,
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_aga_statistics(self, model_name: str) -> Dict[str, Any]:
        """Get AGA statistics for a model"""
        aga = self.aga_modules.get(model_name)
        if not aga:
            return {"error": "AGA not initialized"}
        
        return aga.get_statistics()
    
    def get_knowledge_list(self, model_name: str) -> List[Dict[str, Any]]:
        """Get list of all knowledge in AGA"""
        aga = self.aga_modules.get(model_name)
        if not aga:
            return []
        
        knowledge = aga.get_active_knowledge()
        return [
            {
                'slot_idx': k.slot_idx,
                'lu_id': k.lu_id,
                'condition': k.condition,
                'decision': k.decision,
                'lifecycle_state': k.lifecycle_state.value,
                'reliability': k.reliability,
                'hit_count': k.hit_count,
                'created_at': k.created_at.isoformat() if k.created_at else None,
            }
            for k in knowledge
        ]
    
    def collect_experiment_data(
        self,
        model_name: str,
        test_prompts: List[str],
    ) -> Dict[str, Any]:
        """Collect experiment data for paper"""
        results = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "bottleneck_dim": self.config.aga_bottleneck_dim,
                "num_slots": self.config.aga_num_slots,
                "target_layers": self.config.aga_target_layers,
            },
            "aga_stats": self.get_aga_statistics(model_name),
            "knowledge_count": len(self.get_knowledge_list(model_name)),
            "inference_results": [],
        }
        
        for prompt in test_prompts:
            result = self.run_inference(model_name, prompt)
            results["inference_results"].append(result)
        
        # Save results
        output_file = Path(self.config.output_dir) / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        results["output_file"] = str(output_file)
        self.experiment_results.append(results)
        
        return results
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded models"""
        return list(self.models.keys())
    
    def get_db_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        return self.persistence.get_statistics()


# ==================== Flask App ====================

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global manager
manager: Optional[AGAExperimentManager] = None
config: Optional[Config] = None


def login_required(f):
    """Decorator for login protection"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if config and config.auth_enabled:
            if not session.get('authenticated'):
                return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# HTML Templates
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AGA Experiment Tool - Login</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .login-box {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 40px;
            width: 360px;
        }
        h1 { color: #00d4ff; font-size: 24px; margin-bottom: 8px; }
        p { color: #888; font-size: 14px; margin-bottom: 24px; }
        input {
            width: 100%;
            padding: 12px 16px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            color: #fff;
            font-size: 14px;
            margin-bottom: 16px;
        }
        input:focus { outline: none; border-color: #00d4ff; }
        button {
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #00d4ff, #0099cc);
            border: none;
            border-radius: 8px;
            color: #fff;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
        }
        button:hover { opacity: 0.9; }
        .error { color: #ff6b6b; font-size: 13px; margin-bottom: 16px; }
    </style>
</head>
<body>
    <div class="login-box">
        <h1>🔬 AGA Experiment Tool</h1>
        <p>Auxiliary Governed Attention</p>
        {% if error %}<div class="error">{{ error }}</div>{% endif %}
        <form method="post">
            <input type="password" name="password" placeholder="Password" autofocus>
            <button type="submit">Login</button>
        </form>
    </div>
</body>
</html>
"""

MAIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AGA Experiment Tool</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #161b22, #21262d);
            border-bottom: 1px solid #30363d;
            padding: 16px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header h1 { color: #58a6ff; font-size: 20px; }
        .header .status { 
            display: flex; 
            align-items: center; 
            gap: 16px;
            font-size: 13px;
        }
        .status-dot { 
            width: 8px; 
            height: 8px; 
            border-radius: 50%; 
            background: #3fb950;
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 24px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }
        .card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            overflow: hidden;
        }
        .card-header {
            background: #21262d;
            padding: 12px 16px;
            border-bottom: 1px solid #30363d;
            font-weight: 600;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .card-body { padding: 16px; }
        .form-group { margin-bottom: 16px; }
        .form-group label { 
            display: block; 
            margin-bottom: 6px; 
            font-size: 13px;
            color: #8b949e;
        }
        input, select, textarea {
            width: 100%;
            padding: 10px 12px;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            font-size: 14px;
        }
        input:focus, select:focus, textarea:focus { 
            outline: none; 
            border-color: #58a6ff;
        }
        textarea { resize: vertical; min-height: 80px; }
        .btn {
            padding: 10px 16px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .btn-primary { background: #238636; color: #fff; }
        .btn-secondary { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; }
        .btn-danger { background: #da3633; color: #fff; }
        .btn:hover { opacity: 0.9; }
        .btn-group { display: flex; gap: 8px; flex-wrap: wrap; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }
        .stat-item {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 12px;
            text-align: center;
        }
        .stat-value { font-size: 24px; font-weight: 700; color: #58a6ff; }
        .stat-label { font-size: 12px; color: #8b949e; margin-top: 4px; }
        .log-output {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 12px;
            font-family: monospace;
            font-size: 12px;
            max-height: 300px;
            overflow-y: auto;
            white-space: pre-wrap;
        }
        .log-entry { margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #21262d; }
        .log-time { color: #8b949e; }
        .log-success { color: #3fb950; }
        .log-error { color: #f85149; }
        .full-width { grid-column: 1 / -1; }
        .knowledge-list { max-height: 250px; overflow-y: auto; }
        .knowledge-item {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 8px;
            font-size: 13px;
        }
        .knowledge-item .condition { color: #58a6ff; }
        .knowledge-item .decision { color: #3fb950; }
        .knowledge-item .meta { color: #8b949e; font-size: 11px; margin-top: 4px; }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
        }
        .badge-probationary { background: #9e6a03; color: #fff; }
        .badge-confirmed { background: #238636; color: #fff; }
        .badge-deprecated { background: #6e7681; color: #fff; }
        .badge-quarantined { background: #da3633; color: #fff; }
        .knowledge-actions { margin-top: 8px; }
        .knowledge-actions button { padding: 4px 8px; font-size: 11px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 AGA Experiment Tool</h1>
        <div class="status">
            <span>Model: <strong id="currentModel">{{ current_model or 'None' }}</strong></span>
            <span><span class="status-dot"></span> Connected</span>
            <a href="/logout" style="color: #8b949e; text-decoration: none;">Logout</a>
        </div>
    </div>
    
    <div class="container">
        <!-- Model Selection -->
        <div class="card">
            <div class="card-header">📦 Model Selection</div>
            <div class="card-body">
                <div class="form-group">
                    <label>Select Model</label>
                    <select id="modelSelect">
                        <optgroup label="Small Models (Demo)">
                            <option value="gpt2">GPT-2 (124M)</option>
                            <option value="gpt2-medium">GPT-2 Medium (355M)</option>
                            <option value="qwen2-0.5b">Qwen2 0.5B</option>
                        </optgroup>
                        <optgroup label="Medium Models">
                            <option value="deepseek-coder-1.3b">DeepSeek Coder 1.3B</option>
                            <option value="qwen-1.8b">Qwen 1.8B</option>
                            <option value="llama-3.2-1b">LLaMA 3.2 1B</option>
                        </optgroup>
                        <optgroup label="Large Models (GPU Required)">
                            <option value="llama-2-7b">LLaMA-2-7B</option>
                            <option value="mistral-7b">Mistral-7B</option>
                        </optgroup>
                    </select>
                </div>
                <div class="btn-group">
                    <button class="btn btn-primary" onclick="loadModel()">Load Model</button>
                    <button class="btn btn-secondary" onclick="refreshStats()">Refresh</button>
                </div>
            </div>
        </div>
        
        <!-- AGA Statistics -->
        <div class="card">
            <div class="card-header">📊 AGA Statistics</div>
            <div class="card-body">
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="statTotalSlots">0</div>
                        <div class="stat-label">Total Slots</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="statActiveSlots">0</div>
                        <div class="stat-label">Active</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="statConfirmed">0</div>
                        <div class="stat-label">Confirmed</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="statHits">0</div>
                        <div class="stat-label">Total Hits</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Knowledge Injection -->
        <div class="card">
            <div class="card-header">💉 Knowledge Injection</div>
            <div class="card-body">
                <div class="form-group">
                    <label>Condition (When to activate)</label>
                    <input type="text" id="condition" placeholder="e.g., capital of France">
                </div>
                <div class="form-group">
                    <label>Decision (What to add)</label>
                    <input type="text" id="decision" placeholder="e.g., Paris">
                </div>
                <div class="form-group">
                    <label>Lifecycle State</label>
                    <select id="lifecycle">
                        <option value="probationary">Probationary (r=0.3)</option>
                        <option value="confirmed">Confirmed (r=1.0)</option>
                    </select>
                </div>
                <button class="btn btn-primary" onclick="injectKnowledge()">Inject Knowledge</button>
            </div>
        </div>
        
        <!-- Knowledge List -->
        <div class="card">
            <div class="card-header">📚 Injected Knowledge</div>
            <div class="card-body">
                <div class="knowledge-list" id="knowledgeList">
                    <div style="color: #8b949e; text-align: center; padding: 20px;">
                        No knowledge injected yet
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Inference Test -->
        <div class="card full-width">
            <div class="card-header">🧪 Inference Test</div>
            <div class="card-body" style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                <div>
                    <div class="form-group">
                        <label>Prompt</label>
                        <textarea id="prompt" placeholder="Enter your prompt..." rows="4"></textarea>
                    </div>
                    <div class="form-group">
                        <label>Max New Tokens</label>
                        <input type="number" id="maxTokens" value="50">
                    </div>
                    <button class="btn btn-primary" onclick="runInference()">Run Inference</button>
                </div>
                <div>
                    <label style="display: block; margin-bottom: 6px; font-size: 13px; color: #8b949e;">Response</label>
                    <div class="log-output" id="inferenceOutput" style="min-height: 150px;">Response will appear here...</div>
                </div>
            </div>
        </div>
        
        <!-- Experiment Data Collection -->
        <div class="card full-width">
            <div class="card-header">📋 Experiment Data Collection</div>
            <div class="card-body">
                <div class="form-group">
                    <label>Test Prompts (one per line)</label>
                    <textarea id="testPrompts" rows="3" placeholder="What is the capital of France?
Who wrote Romeo and Juliet?
What is 2 + 2?"></textarea>
                </div>
                <div class="btn-group">
                    <button class="btn btn-primary" onclick="collectData()">Collect Data</button>
                    <button class="btn btn-secondary" onclick="downloadResults()">Download Results</button>
                    <button class="btn btn-secondary" onclick="showDbStats()">DB Stats</button>
                </div>
            </div>
        </div>
        
        <!-- Activity Log -->
        <div class="card full-width">
            <div class="card-header">📝 Activity Log</div>
            <div class="card-body">
                <div class="log-output" id="activityLog">Waiting for activity...</div>
            </div>
        </div>
    </div>
    
    <script>
        let currentModel = '{{ current_model or "" }}';
        
        function log(message, type = 'info') {
            const logDiv = document.getElementById('activityLog');
            const time = new Date().toLocaleTimeString();
            const className = type === 'success' ? 'log-success' : (type === 'error' ? 'log-error' : '');
            logDiv.innerHTML = `<div class="log-entry"><span class="log-time">[${time}]</span> <span class="${className}">${message}</span></div>` + logDiv.innerHTML;
        }
        
        async function loadModel() {
            const model = document.getElementById('modelSelect').value;
            log(`Loading model: ${model}...`);
            
            try {
                const response = await fetch('/api/load_model', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({model: model})
                });
                const data = await response.json();
                
                if (data.success) {
                    currentModel = model;
                    document.getElementById('currentModel').textContent = model;
                    log(`Model ${model} loaded successfully`, 'success');
                    refreshStats();
                    refreshKnowledge();
                } else {
                    log(`Failed to load model: ${data.error}`, 'error');
                }
            } catch (e) {
                log(`Error: ${e.message}`, 'error');
            }
        }
        
        async function refreshStats() {
            if (!currentModel) return;
            
            try {
                const response = await fetch(`/api/stats?model=${currentModel}`);
                const data = await response.json();
                
                if (data.total_slots !== undefined) {
                    document.getElementById('statTotalSlots').textContent = data.total_slots;
                    document.getElementById('statActiveSlots').textContent = data.active_slots || 0;
                    const dist = data.state_distribution || {};
                    document.getElementById('statConfirmed').textContent = dist.confirmed || 0;
                    document.getElementById('statHits').textContent = data.total_hits || 0;
                }
            } catch (e) {
                console.error('Failed to refresh stats:', e);
            }
        }
        
        async function refreshKnowledge() {
            if (!currentModel) return;
            
            try {
                const response = await fetch(`/api/knowledge?model=${currentModel}`);
                const data = await response.json();
                
                const listDiv = document.getElementById('knowledgeList');
                
                if (data.length === 0) {
                    listDiv.innerHTML = '<div style="color: #8b949e; text-align: center; padding: 20px;">No knowledge injected yet</div>';
                    return;
                }
                
                listDiv.innerHTML = data.map(k => `
                    <div class="knowledge-item">
                        <div><span class="condition">${k.condition || 'N/A'}</span> → <span class="decision">${k.decision || 'N/A'}</span></div>
                        <div class="meta">
                            <span class="badge badge-${k.lifecycle_state}">${k.lifecycle_state}</span>
                            LU: ${k.lu_id} | Slot: ${k.slot_idx} | Hits: ${k.hit_count}
                        </div>
                        <div class="knowledge-actions">
                            <button class="btn btn-secondary" onclick="confirmKnowledge('${k.lu_id}')">Confirm</button>
                            <button class="btn btn-danger" onclick="quarantineKnowledge('${k.lu_id}')">Quarantine</button>
                        </div>
                    </div>
                `).join('');
            } catch (e) {
                console.error('Failed to refresh knowledge:', e);
            }
        }
        
        async function injectKnowledge() {
            if (!currentModel) {
                log('Please load a model first', 'error');
                return;
            }
            
            const condition = document.getElementById('condition').value;
            const decision = document.getElementById('decision').value;
            const lifecycle = document.getElementById('lifecycle').value;
            
            if (!condition || !decision) {
                log('Please fill in both condition and decision', 'error');
                return;
            }
            
            log(`Injecting knowledge: "${condition}" → "${decision}"...`);
            
            try {
                const response = await fetch('/api/inject', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        model: currentModel,
                        condition: condition,
                        decision: decision,
                        lifecycle_state: lifecycle
                    })
                });
                const data = await response.json();
                
                if (data.success) {
                    log(`Knowledge injected: ${data.lu_id} at slot ${data.slot_idx}`, 'success');
                    refreshStats();
                    refreshKnowledge();
                    document.getElementById('condition').value = '';
                    document.getElementById('decision').value = '';
                } else {
                    log(`Injection failed: ${data.error}`, 'error');
                }
            } catch (e) {
                log(`Error: ${e.message}`, 'error');
            }
        }
        
        async function confirmKnowledge(luId) {
            try {
                const response = await fetch('/api/lifecycle', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        model: currentModel,
                        lu_id: luId,
                        new_state: 'confirmed'
                    })
                });
                const data = await response.json();
                if (data.success) {
                    log(`Knowledge ${luId} confirmed`, 'success');
                    refreshStats();
                    refreshKnowledge();
                }
            } catch (e) {
                log(`Error: ${e.message}`, 'error');
            }
        }
        
        async function quarantineKnowledge(luId) {
            if (!confirm('Quarantine this knowledge? It will be immediately removed from inference.')) return;
            
            try {
                const response = await fetch('/api/quarantine', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        model: currentModel,
                        lu_id: luId
                    })
                });
                const data = await response.json();
                if (data.success) {
                    log(`Knowledge ${luId} quarantined`, 'success');
                    refreshStats();
                    refreshKnowledge();
                }
            } catch (e) {
                log(`Error: ${e.message}`, 'error');
            }
        }
        
        async function runInference() {
            if (!currentModel) {
                log('Please load a model first', 'error');
                return;
            }
            
            const prompt = document.getElementById('prompt').value;
            const maxTokens = parseInt(document.getElementById('maxTokens').value);
            
            if (!prompt) {
                log('Please enter a prompt', 'error');
                return;
            }
            
            log(`Running inference...`);
            document.getElementById('inferenceOutput').textContent = 'Processing...';
            
            try {
                const response = await fetch('/api/inference', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        model: currentModel,
                        prompt: prompt,
                        max_new_tokens: maxTokens
                    })
                });
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('inferenceOutput').textContent = data.response;
                    log('Inference completed', 'success');
                } else {
                    document.getElementById('inferenceOutput').textContent = `Error: ${data.error}`;
                    log(`Inference failed: ${data.error}`, 'error');
                }
            } catch (e) {
                log(`Error: ${e.message}`, 'error');
            }
        }
        
        async function collectData() {
            if (!currentModel) {
                log('Please load a model first', 'error');
                return;
            }
            
            const prompts = document.getElementById('testPrompts').value.split('\\n').filter(p => p.trim());
            
            if (prompts.length === 0) {
                log('Please enter at least one test prompt', 'error');
                return;
            }
            
            log(`Collecting experiment data with ${prompts.length} prompts...`);
            
            try {
                const response = await fetch('/api/collect', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        model: currentModel,
                        prompts: prompts
                    })
                });
                const data = await response.json();
                
                if (data.output_file) {
                    log(`Data collected and saved to: ${data.output_file}`, 'success');
                } else {
                    log('Data collection completed', 'success');
                }
            } catch (e) {
                log(`Error: ${e.message}`, 'error');
            }
        }
        
        function downloadResults() {
            window.open('/api/download_results', '_blank');
        }
        
        async function showDbStats() {
            try {
                const response = await fetch('/api/db_stats');
                const data = await response.json();
                log(`DB Stats: ${JSON.stringify(data)}`, 'info');
            } catch (e) {
                log(`Error: ${e.message}`, 'error');
            }
        }
        
        // Refresh stats periodically
        setInterval(() => {
            refreshStats();
            refreshKnowledge();
        }, 30000);
    </script>
</body>
</html>
"""


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        password = request.form.get('password', '')
        if password == config.auth_password:
            session['authenticated'] = True
            return redirect(url_for('index'))
        else:
            error = 'Invalid password'
    return render_template_string(LOGIN_TEMPLATE, error=error)


@app.route('/logout')
def logout():
    session.pop('authenticated', None)
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    loaded_models = manager.get_loaded_models() if manager else []
    current = loaded_models[0] if loaded_models else None
    return render_template_string(MAIN_TEMPLATE, current_model=current)


@app.route('/api/load_model', methods=['POST'])
@login_required
def api_load_model():
    data = request.json
    model_name = data.get('model')
    
    if not model_name:
        return jsonify({"success": False, "error": "No model specified"})
    
    success = manager.load_model(model_name)
    return jsonify({"success": success, "model": model_name})


@app.route('/api/stats')
@login_required
def api_stats():
    model_name = request.args.get('model')
    if not model_name:
        return jsonify({"error": "No model specified"})
    
    stats = manager.get_aga_statistics(model_name)
    return jsonify(stats)


@app.route('/api/knowledge')
@login_required
def api_knowledge():
    model_name = request.args.get('model')
    if not model_name:
        return jsonify([])
    
    knowledge = manager.get_knowledge_list(model_name)
    return jsonify(knowledge)


@app.route('/api/inject', methods=['POST'])
@login_required
def api_inject():
    data = request.json
    result = manager.inject_knowledge(
        model_name=data.get('model'),
        condition=data.get('condition'),
        decision=data.get('decision'),
        lifecycle_state=data.get('lifecycle_state', 'probationary'),
    )
    return jsonify(result)


@app.route('/api/lifecycle', methods=['POST'])
@login_required
def api_lifecycle():
    data = request.json
    result = manager.update_lifecycle(
        model_name=data.get('model'),
        lu_id=data.get('lu_id'),
        new_state=data.get('new_state'),
    )
    return jsonify(result)


@app.route('/api/quarantine', methods=['POST'])
@login_required
def api_quarantine():
    data = request.json
    result = manager.quarantine_knowledge(
        model_name=data.get('model'),
        lu_id=data.get('lu_id'),
    )
    return jsonify(result)


@app.route('/api/inference', methods=['POST'])
@login_required
def api_inference():
    data = request.json
    result = manager.run_inference(
        model_name=data.get('model'),
        prompt=data.get('prompt'),
        max_new_tokens=data.get('max_new_tokens', 50),
    )
    return jsonify(result)


@app.route('/api/collect', methods=['POST'])
@login_required
def api_collect():
    data = request.json
    result = manager.collect_experiment_data(
        model_name=data.get('model'),
        test_prompts=data.get('prompts', []),
    )
    return jsonify(result)


@app.route('/api/download_results')
@login_required
def api_download_results():
    results = manager.experiment_results
    return jsonify(results)


@app.route('/api/db_stats')
@login_required
def api_db_stats():
    stats = manager.get_db_statistics()
    return jsonify(stats)


def main():
    global manager, config
    
    parser = argparse.ArgumentParser(description='AGA Experiment Tool')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--port', type=int, default=None, help='Override port')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent / args.config
    if config_path.exists():
        config = Config.from_yaml(str(config_path))
    else:
        config = Config()
        logger.warning(f"Config file not found: {config_path}, using defaults")
    
    if args.port:
        config.port = args.port
    
    # Initialize manager
    manager = AGAExperimentManager(config)
    
    # Run server
    logger.info(f"Starting AGA Experiment Tool on http://{config.host}:{config.port}")
    logger.info(f"Database: {config.db_path}")
    app.run(host=config.host, port=config.port, debug=config.debug)


if __name__ == '__main__':
    main()
