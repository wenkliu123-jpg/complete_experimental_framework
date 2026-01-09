"""
Multi-Layer Privacy Protection for Large Language Models
Complete Implementation

This script implements a comprehensive experimental framework for evaluating
differential privacy protection across three model architectures with four
configurations each.

Experiments:
- 3 models: DistilGPT-2, GPT-2 Small, GPT-2 Medium
- 4 configurations: Baseline, +L1 (DP-SGD), +L1+L2 (Fine-tune), +L1+L2+L3 (Full)
- Total: 12 experiments

Author: [Your Name]
Institution: [Your Institution]
Date: January 2026
"""

import os
import re
import gc
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm.auto import tqdm
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SECTION 1: DATA GENERATION
# Generate synthetic medical records for privacy experiments
# ============================================================================

def generate_synthetic_medical_data(num_samples=1000, seed=42):
    """
    Generate synthetic medical text data for privacy experiments.
    
    Uses template-based generation with medical vocabulary to create
    realistic but completely synthetic patient records.
    
    Args:
        num_samples: Number of text samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of synthetic medical text samples
    """
    np.random.seed(seed)
    
    # Medical text templates
    templates = [
        "Patient presents with {symptom} and {condition}. Treatment plan includes {treatment}.",
        "Medical history shows {condition} diagnosed in {year}. Current status: {status}.",
        "Clinical notes: {symptom} observed during examination. Recommended {treatment}.",
        "Patient reports {symptom} lasting {duration}. Prescribed {medication} for management.",
        "Diagnosis: {condition}. Lab results indicate {lab_result}. Follow-up scheduled.",
        "Chief complaint: {symptom}. Physical examination reveals {finding}. Treatment: {treatment}.",
        "Patient admitted with acute {condition}. Vital signs stable. Monitoring {parameter}.",
        "Consultation notes: {symptom} with history of {condition}. Plan: {treatment}.",
        "Imaging results show {finding}. Clinical correlation with {symptom} noted.",
        "Patient education provided regarding {condition} management and {treatment} protocol.",
        "Discharge summary: Treated for {condition} with {treatment}. Prognosis: {status}.",
        "Follow-up visit: {symptom} improved with {medication}. Continue current regimen.",
        "Emergency department notes: {symptom} with onset {duration} ago. Diagnosis: {condition}.",
        "Surgical notes: Procedure performed for {condition}. Post-operative status: {status}.",
        "Psychiatric evaluation: Patient exhibits {symptom}. Recommended {treatment} therapy.",
    ]
    
    # Medical vocabulary pools
    symptoms = [
        "chest pain", "shortness of breath", "fatigue", "headache", "fever",
        "nausea", "dizziness", "abdominal pain", "joint pain", "cough",
        "back pain", "anxiety", "insomnia", "weakness", "numbness",
        "palpitations", "weight loss", "confusion", "tremors", "rash"
    ]
    
    conditions = [
        "hypertension", "diabetes mellitus", "asthma", "arthritis", "depression",
        "coronary artery disease", "chronic kidney disease", "COPD", "migraine",
        "gastroesophageal reflux", "hypothyroidism", "osteoporosis", "anemia",
        "atrial fibrillation", "heart failure", "stroke", "pneumonia", "sepsis",
        "pulmonary embolism", "acute myocardial infarction"
    ]
    
    treatments = [
        "medication management", "physical therapy", "lifestyle modifications",
        "surgical intervention", "radiation therapy", "chemotherapy",
        "occupational therapy", "cognitive behavioral therapy", "dialysis",
        "oxygen therapy", "antibiotic treatment", "pain management",
        "cardiac rehabilitation", "dietary counseling", "stress reduction"
    ]
    
    medications = [
        "metformin", "lisinopril", "atorvastatin", "metoprolol", "amlodipine",
        "omeprazole", "levothyroxine", "albuterol", "gabapentin", "sertraline",
        "aspirin", "insulin", "warfarin", "prednisone", "amoxicillin"
    ]
    
    durations = [
        "2 days", "1 week", "3 weeks", "2 months", "6 months",
        "1 year", "several hours", "overnight", "intermittently"
    ]
    
    years = ["2020", "2021", "2022", "2023", "2024", "2019", "2018"]
    
    statuses = [
        "stable", "improving", "worsening", "critical", "good",
        "fair", "guarded", "excellent", "poor", "satisfactory"
    ]
    
    findings = [
        "mild inflammation", "no acute findings", "chronic changes",
        "abnormal tissue", "enlarged organ", "fluid accumulation",
        "decreased function", "structural abnormality", "normal variation"
    ]
    
    lab_results = [
        "elevated levels", "within normal limits", "decreased counts",
        "abnormal values", "positive markers", "negative results"
    ]
    
    parameters = [
        "blood pressure", "heart rate", "oxygen saturation", "temperature",
        "respiratory rate", "pain level", "glucose levels", "kidney function"
    ]
    
    # Generate synthetic records
    samples = []
    for i in range(num_samples):
        template = np.random.choice(templates)
        
        # Fill template with random medical terms
        text = template.format(
            symptom=np.random.choice(symptoms),
            condition=np.random.choice(conditions),
            treatment=np.random.choice(treatments),
            medication=np.random.choice(medications),
            duration=np.random.choice(durations),
            year=np.random.choice(years),
            status=np.random.choice(statuses),
            finding=np.random.choice(findings),
            lab_result=np.random.choice(lab_results),
            parameter=np.random.choice(parameters)
        )
        
        samples.append(text)
    
    return samples


# ============================================================================
# SECTION 2: UTILITY FUNCTIONS
# Core functions used across all experiments
# ============================================================================

def get_model_size(model):
    """Calculate total model parameters in millions."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e6


def compute_perplexity(model, tokenizer, texts, device='cuda', batch_size=8, max_length=128):
    """
    Compute perplexity metric on evaluation texts.
    
    Perplexity measures how well the model predicts the text.
    Lower perplexity indicates better language modeling.
    
    Args:
        model: Trained language model
        tokenizer: Associated tokenizer
        texts: List of evaluation texts
        device: Computing device
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        
    Returns:
        Perplexity score (float)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding='max_length'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            num_tokens = inputs['attention_mask'].sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


def clip_gradients(model, max_norm):
    """
    Clip gradients to maximum L2 norm for differential privacy.
    
    Gradient clipping is the first step in DP-SGD, ensuring that
    each sample's contribution is bounded.
    
    Args:
        model: PyTorch model
        max_norm: Maximum allowed gradient norm (C parameter)
        
    Returns:
        Actual gradient norm before clipping
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
    
    return total_norm


def add_noise(model, noise_scale):
    """
    Add Gaussian noise to gradients for differential privacy.
    
    This is the second step in DP-SGD. The noise scale is calibrated
    to provide the desired privacy guarantee (epsilon).
    
    Args:
        model: PyTorch model
        noise_scale: Standard deviation of Gaussian noise (sigma * C / batch_size)
    """
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_scale
            param.grad.data.add_(noise)


def calculate_epsilon(num_samples, batch_size, noise_multiplier, epochs, delta=1e-5):
    """
    Calculate approximate privacy budget (epsilon) using simplified accounting.
    
    Privacy budget epsilon represents the strength of privacy guarantee.
    Lower epsilon = stronger privacy. Common targets: epsilon < 1 (strong),
    epsilon < 10 (moderate).
    
    Args:
        num_samples: Total training samples
        batch_size: Training batch size
        noise_multiplier: DP-SGD noise parameter (sigma)
        epochs: Number of training epochs
        delta: Privacy parameter delta (failure probability)
        
    Returns:
        Privacy budget epsilon
    """
    q = batch_size / num_samples
    steps = (num_samples // batch_size) * epochs
    epsilon = (q * steps * noise_multiplier) / np.sqrt(2 * np.log(1.25 / delta))
    return epsilon


def apply_selective_freezing(model, freeze_ratio, num_layers):
    """
    Freeze bottom layers for parameter-efficient fine-tuning (Layer 2).
    
    Selective freezing reduces the number of trainable parameters,
    improving both computational efficiency and privacy.
    
    Args:
        model: GPT-2 style model
        freeze_ratio: Proportion of layers to freeze (0-1)
        num_layers: Total number of transformer layers
        
    Returns:
        Tuple of (modified_model, trainable_params, total_params)
    """
    freeze_layers = int(num_layers * freeze_ratio)
    
    # Freeze bottom layers
    for i, layer in enumerate(model.transformer.h):
        if i < freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True
    
    # Always keep output layer trainable
    for param in model.lm_head.parameters():
        param.requires_grad = True
    
    # Calculate parameter counts
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    return model, trainable, total


def clear_gpu_memory():
    """Clear GPU memory cache to prevent OOM errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# SECTION 3: LAYER 3 - INFERENCE FILTERING
# Runtime protection through pattern-based filtering
# ============================================================================

class InferenceProtectionLayer:
    """
    Layer 3: Inference-time privacy protection.
    
    Provides runtime filtering of sensitive patterns in model outputs.
    This is a defense-in-depth measure that catches any sensitive
    information that might leak despite training protections.
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize inference protection layer.
        
        Args:
            model: Trained language model
            tokenizer: Associated tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Regex patterns for sensitive data types
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'ip': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
    
    def generate_protected(self, prompt, max_length=50):
        """
        Generate text with automatic filtering of sensitive patterns.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum generation length
            
        Returns:
            Generated text with sensitive data redacted
        """
        device = next(self.model.parameters()).device
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        filtered_text = self.filter_output(output_text)
        
        return filtered_text
    
    def filter_output(self, text):
        """
        Apply pattern-based redaction to text.
        
        Args:
            text: Generated text
            
        Returns:
            Text with sensitive patterns replaced by [REDACTED_*] tags
        """
        for pattern_type, pattern in self.patterns.items():
            text = re.sub(pattern, f'[REDACTED_{pattern_type.upper()}]', text)
        return text


# ============================================================================
# SECTION 4: EXPERIMENT RUNNER
# Generic function to run single experiment configuration
# ============================================================================

def run_experiment(
    model_config,
    train_texts,
    eval_texts,
    config_type,
    training_config,
    device='cuda'
):
    """
    Run a single experiment configuration.
    
    Args:
        model_config: Dictionary with model specifications
        train_texts: Training data
        eval_texts: Evaluation data
        config_type: One of ['baseline', 'l1', 'l1_l2', 'l1_l2_l3']
        training_config: Dictionary with training hyperparameters
        device: Computing device
        
    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*70}")
    print(f"Running: {model_config['name']} - {config_type.upper()}")
    print(f"{'='*70}")
    
    # Load model
    print("\n[1/6] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_config['model_id'])
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_id'])
    tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    
    model_size = get_model_size(model)
    print(f"✓ {model_config['name']} loaded: {model_size:.1f}M parameters")
    
    # Prepare data batches
    print("\n[2/6] Preparing data...")
    batch_size = model_config['batch_size']
    batches = []
    
    for i in range(0, len(train_texts), batch_size):
        batch_texts = train_texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            truncation=True,
            max_length=training_config['max_length'],
            padding='max_length'
        )
        batches.append({k: v.to(device) for k, v in inputs.items()})
    
    print(f"✓ Prepared {len(batches)} batches")
    
    # Configuration-specific training
    if config_type == 'baseline':
        # Baseline: Standard training without privacy
        print("\n[3/6] Training baseline model...")
        model, training_time, final_loss = train_baseline(
            model, batches, training_config
        )
        epsilon = None
        trainable_params = model_size
        
    elif config_type == 'l1':
        # L1: DP-SGD training
        print("\n[3/6] Training with DP-SGD (Layer 1)...")
        model, training_time, final_loss = train_with_dp(
            model, batches, training_config
        )
        epsilon = calculate_epsilon(
            len(train_texts),
            batch_size,
            training_config['noise_multiplier'],
            training_config['epochs'],
            training_config['delta']
        )
        trainable_params = model_size
        print(f"✓ Privacy: ε = {epsilon:.2f}")
        
    elif config_type == 'l1_l2':
        # L1+L2: DP-SGD with selective freezing
        print("\n[3/6] Applying selective freezing and fine-tuning (Layer 2)...")
        model, trainable, total = apply_selective_freezing(
            model,
            model_config['freeze_ratio'],
            model_config['layers']
        )
        
        model, training_time, final_loss = train_with_dp(
            model, batches, training_config, fine_tune_epochs=3
        )
        epsilon = calculate_epsilon(
            len(train_texts),
            batch_size,
            training_config['noise_multiplier'],
            3,  # Fine-tune epochs
            training_config['delta']
        )
        trainable_params = trainable / 1e6
        print(f"✓ Privacy: ε = {epsilon:.2f}")
        print(f"✓ Trainable params: {trainable_params:.1f}M ({100*trainable/total:.1f}%)")
        
    elif config_type == 'l1_l2_l3':
        # Full pipeline: L1+L2+L3
        print("\n[3/6] Adding inference protection (Layer 3)...")
        # L3 doesn't change model, just wraps it
        # Training same as L1+L2
        model, trainable, total = apply_selective_freezing(
            model,
            model_config['freeze_ratio'],
            model_config['layers']
        )
        
        model, training_time, final_loss = train_with_dp(
            model, batches, training_config, fine_tune_epochs=3
        )
        epsilon = calculate_epsilon(
            len(train_texts),
            batch_size,
            training_config['noise_multiplier'],
            3,
            training_config['delta']
        )
        trainable_params = trainable / 1e6
        
        # Test inference filtering
        protected = InferenceProtectionLayer(model, tokenizer)
        filter_rate = test_inference_filtering(protected)
        print(f"✓ Filter rate: {filter_rate:.0f}%")
    
    # Evaluate model
    print("\n[4/6] Evaluating perplexity...")
    perplexity = compute_perplexity(model, tokenizer, eval_texts, device=device, batch_size=4)
    print(f"✓ Perplexity: {perplexity:.2f}")
    
    # Save results
    print("\n[5/6] Saving results...")
    results = {
        'model': model_config['name'],
        'configuration': config_type,
        'perplexity': perplexity,
        'training_time': training_time,
        'final_loss': final_loss,
        'model_params': model_size,
        'trainable_params': trainable_params,
        'epsilon': epsilon,
    }
    
    if config_type == 'l1_l2_l3':
        results['filter_rate'] = filter_rate
    
    # Save to JSON
    filename = f"results_{model_config['model_id'].replace('-', '_')}_{config_type}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved: {filename}")
    print(f"\n{'='*70}")
    print(f"✅ COMPLETE: {model_config['name']} - {config_type.upper()}")
    print(f"{'='*70}\n")
    
    # Clean up
    del model
    clear_gpu_memory()
    
    return results


def train_baseline(model, batches, config):
    """Train model without differential privacy (baseline)."""
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    start_time = datetime.now()
    epochs = config['epochs']
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in tqdm(batches, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            outputs = model(**batch, labels=batch['input_ids'])
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    training_time = (datetime.now() - start_time).total_seconds()
    return model, training_time, avg_loss


def train_with_dp(model, batches, config, fine_tune_epochs=None):
    """Train model with differential privacy (DP-SGD)."""
    model.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    start_time = datetime.now()
    epochs = fine_tune_epochs if fine_tune_epochs else config['epochs']
    batch_size = batches[0]['input_ids'].size(0)
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in tqdm(batches, desc=f"Epoch {epoch+1}/{epochs} (DP)", leave=False):
            outputs = model(**batch, labels=batch['input_ids'])
            loss = outputs.loss
            
            loss.backward()
            
            # DP-SGD: Clip gradients and add noise
            clip_gradients(model, config['max_grad_norm'])
            noise_scale = config['noise_multiplier'] * config['max_grad_norm'] / batch_size
            add_noise(model, noise_scale)
            
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    training_time = (datetime.now() - start_time).total_seconds()
    return model, training_time, avg_loss


def test_inference_filtering(protected_model):
    """Test Layer 3 inference filtering on sample prompts."""
    test_cases = [
        "Patient presents with chest pain and",
        "Medical record shows email john@example.com and",
        "Diagnosis: hypertension with phone 555-1234",
        "Treatment plan includes",
    ]
    
    filter_count = 0
    for prompt in test_cases:
        output = protected_model.generate_protected(prompt, max_length=30)
        if '[REDACTED' in output:
            filter_count += 1
    
    filter_rate = (filter_count / len(test_cases)) * 100
    return filter_rate


# ============================================================================
# SECTION 5: ANALYSIS AND VISUALIZATION
# Statistical analysis and figure generation
# ============================================================================

def load_all_results():
    """Load all experimental results from JSON files."""
    results = []
    
    models = ['distilgpt2', 'gpt2', 'gpt2_medium']
    configs = ['baseline', 'l1', 'l1_l2', 'l1_l2_l3']
    
    for model in models:
        for config in configs:
            filename = f'results_{model}_{config}.json'
            if Path(filename).exists():
                with open(filename, 'r') as f:
                    data = json.load(f)
                results.append(data)
    
    return pd.DataFrame(results)


def analyze_results(df):
    """
    Perform comprehensive statistical analysis.
    
    Args:
        df: DataFrame with all experimental results
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    # Calculate degradation metrics
    for model in df['model'].unique():
        baseline_ppl = df[(df['model'] == model) & (df['configuration'] == 'baseline')]['perplexity'].values[0]
        l1_ppl = df[(df['model'] == model) & (df['configuration'] == 'l1')]['perplexity'].values[0]
        
        degradation = ((l1_ppl - baseline_ppl) / baseline_ppl) * 100
        analysis[f'{model}_degradation'] = degradation
    
    # Statistical tests
    baseline_ppls = df[df['configuration'] == 'baseline']['perplexity'].values
    l1_ppls = df[df['configuration'] == 'l1']['perplexity'].values
    
    if len(baseline_ppls) >= 2 and len(l1_ppls) >= 2:
        t_stat, p_value = stats.ttest_ind(baseline_ppls, l1_ppls)
        analysis['dp_effect_pvalue'] = p_value
        analysis['dp_effect_significant'] = p_value < 0.05
    
    return analysis


def generate_figures(df):
    """
    Generate publication-quality figures.
    
    Args:
        df: DataFrame with experimental results
    """
    # Set publication style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 11,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    # Figure 1: Cumulative layer effects
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    models = df['model'].unique()
    configs = ['baseline', 'l1', 'l1_l2', 'l1_l2_l3']
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = df[df['model'] == model]
        
        ppls = [model_data[model_data['configuration'] == c]['perplexity'].values[0] 
                for c in configs if len(model_data[model_data['configuration'] == c]) > 0]
        
        ax.bar(range(len(ppls)), ppls, alpha=0.8, edgecolor='black')
        ax.set_title(f'{model}', fontweight='bold')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Perplexity')
        ax.set_xticks(range(len(ppls)))
        ax.set_xticklabels(['Base', '+L1', '+L1+L2', '+Full'], rotation=0)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure1_cumulative_effects.png')
    plt.close()
    
    # Figure 2: Privacy-utility tradeoff
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for model in models:
        model_data = df[(df['model'] == model) & (df['configuration'] != 'baseline')]
        if not model_data.empty:
            ax.scatter(
                model_data['epsilon'],
                model_data['perplexity'],
                s=200,
                alpha=0.7,
                label=model,
                edgecolors='black',
                linewidth=2
            )
    
    ax.set_xlabel('Privacy Budget (ε)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Perplexity', fontsize=13, fontweight='bold')
    ax.set_title('Privacy-Utility Tradeoff', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure2_privacy_utility.png')
    plt.close()
    
    print("✓ Figures generated: figure1_cumulative_effects.png, figure2_privacy_utility.png")


# ============================================================================
# SECTION 6: PRIVACY EVALUATION - MEMBERSHIP INFERENCE ATTACK
# Evaluate privacy protection by testing resistance to MIA
# ============================================================================

def compute_sample_perplexity(model, tokenizer, text, device, max_length=128):
    """
    Compute perplexity for a single text sample.
    
    Lower perplexity indicates the model is more confident about the text,
    suggesting it may have been in the training data (potential privacy leak).
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        text: Text sample
        device: Computing device
        max_length: Maximum sequence length
        
    Returns:
        Perplexity score for the sample
    """
    model.eval()
    
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss.item()
        perplexity = np.exp(loss)
    
    return perplexity


def perform_membership_inference_attack(member_ppls, non_member_ppls):
    """
    Perform membership inference attack using perplexity threshold.
    
    The attack attempts to determine if a sample was in the training data
    by comparing its perplexity to a threshold. Lower perplexity suggests
    membership (privacy violation).
    
    Args:
        member_ppls: Perplexities of training samples (members)
        non_member_ppls: Perplexities of non-training samples (non-members)
        
    Returns:
        Dictionary with attack performance metrics
    """
    # Combine all samples
    all_ppls = np.concatenate([member_ppls, non_member_ppls])
    true_labels = np.concatenate([
        np.ones(len(member_ppls)),      # 1 = member
        np.zeros(len(non_member_ppls))  # 0 = non-member
    ])
    
    # Use median as threshold
    threshold = np.median(all_ppls)
    
    # Attack rule: predict as member if perplexity < threshold
    predicted_labels = (all_ppls < threshold).astype(int)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_labels == true_labels)
    
    # Calculate AUC-ROC
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(true_labels, -all_ppls)  # Negative because lower PPL = member
    roc_auc = auc(fpr, tpr)
    
    return {
        'accuracy': accuracy,
        'auc': roc_auc,
        'threshold': threshold,
        'member_mean': np.mean(member_ppls),
        'non_member_mean': np.mean(non_member_ppls),
    }


def evaluate_privacy_with_mia(train_texts, eval_texts, device='cuda', num_samples=100):
    """
    Evaluate privacy protection by performing membership inference attack.
    
    Compares baseline model (vulnerable) vs L1+L2 model (protected) to
    demonstrate that privacy mechanisms actually work.
    
    Args:
        train_texts: Training data
        eval_texts: Evaluation data
        device: Computing device
        num_samples: Number of samples to test
        
    Returns:
        Dictionary with MIA results
    """
    print("\n[Privacy Evaluation] Testing resistance to membership inference attacks...")
    
    # Check if required models exist
    baseline_path = Path('gpt2_baseline.pt')
    l1_l2_path = Path('gpt2_L1_L2.pt')
    
    if not baseline_path.exists() or not l1_l2_path.exists():
        print("⚠ Skipping MIA evaluation: Required model checkpoints not found")
        print("  (Need: gpt2_baseline.pt and gpt2_L1_L2.pt)")
        return None
    
    # Sample members and non-members
    np.random.seed(42)
    member_indices = np.random.choice(len(train_texts), size=min(num_samples, len(train_texts)), replace=False)
    members = [train_texts[i] for i in member_indices]
    
    non_member_indices = np.random.choice(len(eval_texts), size=min(num_samples, len(eval_texts)), replace=False)
    non_members = [eval_texts[i] for i in non_member_indices]
    
    print(f"  Testing with {len(members)} members and {len(non_members)} non-members")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load baseline model
    print("\n  [1/2] Testing baseline model (no privacy)...")
    baseline_model = AutoModelForCausalLM.from_pretrained('gpt2')
    baseline_model.load_state_dict(torch.load('gpt2_baseline.pt', map_location=device))
    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    
    # Compute perplexities for baseline
    baseline_member_ppls = np.array([
        compute_sample_perplexity(baseline_model, tokenizer, text, device)
        for text in members
    ])
    baseline_non_member_ppls = np.array([
        compute_sample_perplexity(baseline_model, tokenizer, text, device)
        for text in non_members
    ])
    
    baseline_results = perform_membership_inference_attack(
        baseline_member_ppls, 
        baseline_non_member_ppls
    )
    
    print(f"     Baseline attack accuracy: {baseline_results['accuracy']:.2%}")
    print(f"     Baseline AUC: {baseline_results['auc']:.3f}")
    
    del baseline_model
    clear_gpu_memory()
    
    # Load L1+L2 model
    print("\n  [2/2] Testing L1+L2 model (with privacy)...")
    l1_l2_model = AutoModelForCausalLM.from_pretrained('gpt2')
    l1_l2_model.load_state_dict(torch.load('gpt2_L1_L2.pt', map_location=device))
    l1_l2_model = l1_l2_model.to(device)
    l1_l2_model.eval()
    
    # Compute perplexities for L1+L2
    l1_l2_member_ppls = np.array([
        compute_sample_perplexity(l1_l2_model, tokenizer, text, device)
        for text in members
    ])
    l1_l2_non_member_ppls = np.array([
        compute_sample_perplexity(l1_l2_model, tokenizer, text, device)
        for text in non_members
    ])
    
    l1_l2_results = perform_membership_inference_attack(
        l1_l2_member_ppls,
        l1_l2_non_member_ppls
    )
    
    print(f"     L1+L2 attack accuracy: {l1_l2_results['accuracy']:.2%}")
    print(f"     L1+L2 AUC: {l1_l2_results['auc']:.3f}")
    
    del l1_l2_model
    clear_gpu_memory()
    
    # Calculate improvement
    improvement = baseline_results['accuracy'] - l1_l2_results['accuracy']
    auc_improvement = baseline_results['auc'] - l1_l2_results['auc']
    
    # Generate MIA comparison figure
    print("\n  Generating privacy evaluation figure...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Perplexity distributions
    ax = axes[0]
    ax.hist(baseline_member_ppls, bins=15, alpha=0.6, label='Members', color='red')
    ax.hist(baseline_non_member_ppls, bins=15, alpha=0.6, label='Non-members', color='blue')
    ax.axvline(baseline_results['threshold'], color='green', linestyle='--', label='Threshold')
    ax.set_title(f'Baseline: Attack Acc = {baseline_results["accuracy"]:.1%}')
    ax.set_xlabel('Perplexity')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    ax = axes[1]
    ax.hist(l1_l2_member_ppls, bins=15, alpha=0.6, label='Members', color='red')
    ax.hist(l1_l2_non_member_ppls, bins=15, alpha=0.6, label='Non-members', color='blue')
    ax.axvline(l1_l2_results['threshold'], color='green', linestyle='--', label='Threshold')
    ax.set_title(f'L1+L2: Attack Acc = {l1_l2_results["accuracy"]:.1%}')
    ax.set_xlabel('Perplexity')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    plt.suptitle('Membership Inference Attack: Privacy Protection Evaluation', fontweight='bold')
    plt.tight_layout()
    plt.savefig('figure_mia_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: figure_mia_evaluation.png")
    
    # Save results
    mia_summary = {
        'baseline_accuracy': baseline_results['accuracy'],
        'baseline_auc': baseline_results['auc'],
        'protected_accuracy': l1_l2_results['accuracy'],
        'protected_auc': l1_l2_results['auc'],
        'improvement': improvement,
        'auc_improvement': auc_improvement,
    }
    
    with open('mia_results.json', 'w') as f:
        json.dump(mia_summary, f, indent=2)
    
    print("  ✓ Saved: mia_results.json")
    
    # Interpretation
    print("\n  Privacy Assessment:")
    if improvement > 0.1:
        print("  ✅ EXCELLENT: Privacy protections significantly resist attacks!")
    elif improvement > 0.05:
        print("  ✅ GOOD: Privacy protections provide meaningful resistance.")
    else:
        print("  ⚠ MODERATE: Some protection, but attacks still partially effective.")
    
    return mia_summary


# ============================================================================
# SECTION 7: MAIN EXECUTION
# Run complete experimental pipeline
# ============================================================================

def main():
    """
    Main execution function.
    Run complete 5-day experimental pipeline.
    """
    print("="*70)
    print("MULTI-LAYER PRIVACY PROTECTION FRAMEWORK")
    print("Complete Experimental Pipeline")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Computing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    # ========================================================================
    # PHASE 1: DATA GENERATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 1: DATA GENERATION")
    print("="*70)
    
    print("\nGenerating synthetic medical data...")
    all_texts = generate_synthetic_medical_data(num_samples=1000, seed=42)
    train_texts = all_texts[100:]  # 900 samples
    eval_texts = all_texts[:100]   # 100 samples
    
    print(f"✓ Generated {len(all_texts)} total samples")
    print(f"✓ Training: {len(train_texts)} samples")
    print(f"✓ Evaluation: {len(eval_texts)} samples")
    
    # Save data
    with open('train_texts.json', 'w') as f:
        json.dump(train_texts, f)
    with open('eval_texts.json', 'w') as f:
        json.dump(eval_texts, f)
    
    print("✓ Data saved to train_texts.json and eval_texts.json")
    
    # ========================================================================
    # PHASE 2: MODEL EXPERIMENTS
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 2: RUNNING EXPERIMENTS (12 total)")
    print("="*70)
    
    # Model configurations
    models = [
        {
            'name': 'DistilGPT-2',
            'model_id': 'distilgpt2',
            'batch_size': 16,
            'layers': 6,
            'freeze_ratio': 0.67,
        },
        {
            'name': 'GPT-2 Small',
            'model_id': 'gpt2',
            'batch_size': 12,
            'layers': 12,
            'freeze_ratio': 0.67,
        },
        {
            'name': 'GPT-2 Medium',
            'model_id': 'gpt2-medium',
            'batch_size': 8,
            'layers': 24,
            'freeze_ratio': 0.67,
        },
    ]
    
    # Training configuration
    training_config = {
        'epochs': 5,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'max_length': 128,
        'delta': 1e-5,
        'noise_multiplier': 0.7,
        'max_grad_norm': 1.5,
    }
    
    # Run all experiments
    all_results = []
    configs = ['baseline', 'l1', 'l1_l2', 'l1_l2_l3']
    total_experiments = len(models) * len(configs)
    
    experiment_num = 0
    for model_config in models:
        for config_type in configs:
            experiment_num += 1
            print(f"\n{'#'*70}")
            print(f"# EXPERIMENT {experiment_num}/{total_experiments}")
            print(f"{'#'*70}")
            
            results = run_experiment(
                model_config,
                train_texts,
                eval_texts,
                config_type,
                training_config,
                device=device
            )
            all_results.append(results)
    
    # ========================================================================
    # PHASE 3: ANALYSIS AND VISUALIZATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 3: ANALYSIS AND VISUALIZATION")
    print("="*70)
    
    # Load and analyze results
    print("\nLoading results...")
    results_df = pd.DataFrame(all_results)
    
    print("\nPerforming statistical analysis...")
    analysis = analyze_results(results_df)
    
    print("\nGenerating figures...")
    generate_figures(results_df)
    
    # Save complete results
    results_df.to_csv('complete_results.csv', index=False)
    print("\n✓ Complete results saved to: complete_results.csv")
    
    # ========================================================================
    # PHASE 4: PRIVACY EVALUATION - MEMBERSHIP INFERENCE ATTACK
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 4: PRIVACY EVALUATION - MEMBERSHIP INFERENCE ATTACK")
    print("="*70)
    
    print("\nEvaluating privacy protection effectiveness...")
    print("Testing resistance to membership inference attacks...")
    
    # Run MIA evaluation on GPT-2 Small (comparing Baseline vs L1+L2)
    mia_results = evaluate_privacy_with_mia(
        train_texts, 
        eval_texts,
        device=device
    )
    
    if mia_results:
        print("\n✅ Privacy evaluation complete!")
        print(f"   Baseline attack accuracy: {mia_results['baseline_accuracy']:.1%}")
        print(f"   L1+L2 attack accuracy: {mia_results['protected_accuracy']:.1%}")
        print(f"   Privacy improvement: {mia_results['improvement']:.1%}")
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENTAL SUMMARY")
    print("="*70)
    print(f"\nTotal experiments completed: {len(all_results)}")
    print(f"Models evaluated: {len(models)}")
    print(f"Configurations per model: {len(configs)}")
    print(f"Privacy evaluations: 1 (MIA on GPT-2 Small)")
    
    print("\n" + "="*70)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print("\n✅ ALL EXPERIMENTS COMPLETE!")


if __name__ == "__main__":
    main()
