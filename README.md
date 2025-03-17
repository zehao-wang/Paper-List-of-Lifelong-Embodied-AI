# Paper List of Knowledge Evolution For Lifelong Embodied AI

We introduce the **Knowledge Evolution for Lifelong Embodied AI** to help position existing research within this ultimate goal and identify key gaps that remain. We welcome your valuable feedback and suggestions to enhance the discussion on lifelong embodied AI. If you have ideas to share, please feel free to raise issues, submit pull requests, or reach out to us directly at zekewang@outlook.com.

Thank you for your support and collaboration!


## Table of Contents
- [Introduction](#introduction)
- [Embodied Agent Initialization](#embodied-agent-initialization)
- [Data Collection](#data-collection)
  - [Human-Curated Data](#human-curated-data)
  - [Active Data Collection](#active-data-collection)
- [Knowledge Consolidation](#knowledge-consolidation)
  - [In-parameter Knowledge Consolidation](#in-parameter-knowledge-consolidation)
  - [In-context Knowledge Consolidation](#in-context-knowledge-consolidation)
- [Knowledge Refinement](#knowledge-refinement)
  - [Space and Time Efficiency](#space-and-time-efficiency)
  - [Generalization](#generalization)

## Introduction
Lifelong embodied AI aims to enable autonomous agents to evolve continuously by acquiring, consolidating, and refining knowledge over time. This repository follows a structured approach to represent the major components of lifelong embodied AI.

## Embodied Agent Initialization
### Agent style models
- Code as policies: Language model programs for embodied control
- Ok-robot: What really matters in integrating open-knowledge models for robotics
- Instruct2act: Mapping multi-modality instructions to robotic actions with large language model
- Voxposer: Composable 3d value maps for robotic manipulation with language models
  
### Embodied Foundation Models/ Vision-Language-Action models
- PaLM-E: An Embodied Multimodal Language Model
- SayCan: Do as I Can, Not as I Say: Grounding Language in Robotic Affordances
- RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
- Open X-Embodiment: Robotic Learning Datasets and RT-X Models
- OpenVLA: An Open-Source Vision-Language-Action Model

## Data Collection
### Human-Curated Data
A breif collection of Embodied AI tasks, more sophisticated can refer to [Embodied AI Paper List](https://github.com/HCPLab-SYSU/Embodied_AI_Paper_List).
- **Robotic Manipulation**:
  - Jacquard: A Large Scale Dataset for Robotic Grasp Detection
  - ACRONYM: A Large-Scale Grasp Dataset Based on Simulation
  - MultigripperGrasp: A Dataset for Robotic Grasping from Parallel Jaw Grippers to Dexterous Hands
- **Embodied Navigation**:
  -  Object-Goal Navigation Using Goal-Oriented Semantic Exploration
  -  DD-PPO: Learning Near-Perfect PointGoal Navigators from 2.5 Billion Frames
  -  Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments
  -  Room-Across-Room: Multilingual Vision-and-Language Navigation with Dense Spatiotemporal Grounding
  -  Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments
- **Embodied QA**:
  - OpenEQA: Embodied Question Answering in the Era of Foundation Models
  - Knowledge-Based Embodied Question Answering
  - Language Models Meet World Models: Embodied Experiences Enhance Language Models

### Active Data Collection

- **Low-level Perceptual Curiosity**: 
  - Learning to explore using active neural slam
  - Semantic curiosity for active visual learning
  - Embodied visual active learning for semantic segmentation
  - Seal: Self-supervised embodied active learning using exploration and 3d consistency
  - Visual curiosity: Learning to ask questions to learn visual recognition
  - Learning to look around: Intelligently exploring unseen environments for unknown tasks
  - The curious robot: Learning visual representations via physical interactions
  - Planning for learning object properties
    
- **Pursuit of High-level Skill Development**:
  - <ins>**Manual-designed policy**</ins>
    - Embodied active learning of relational state abstractions for bilevel planning
    - A framework for learning from demonstration with minimal human effort
    - Active reward learning
    - Shaping imbalance into balance: Active robot guidance of human teachers for better learning from demonstrations
    - Self-supervised exploration via disagreement
    - Intrinsically motivated goal exploration processes with automatic curriculum learning
    - Curious exploration via structured world models yields zero-shot object manipulation
    - Optimistic active exploration of dynamical systems
    - Designing robot learners that ask good questions
    - Pre-trained language models for interactive decision-making
    - Training language models to follow instructions with human feedback
    - Vlp: Vision-language preference learning for embodied manipulation
    - Learning a universal human prior for dexterous manipulation from human preference
    - Skill preferences: Learning to extract and execute robotic skills from human feedback
  - <ins>**LLM-driven policy**</ins>
    - Voyager: An open-ended embodied agent with large language models
    - Expel: Llm agents are experiential learners
    - Jarvis-1: Open-world multi-task agents with memory-augmented multimodal language models
    - Lifelong robot library learning: Bootstrapping composable and generalizable skills for embodied control with language models

## Knowledge Consolidation
### In-parameter Knowledge Consolidation
- **Regularization Approaches**:
  - Language models meet world models: Embodied experiences enhance language models
  - Airloop: Lifelong loop closure detection
  - Continual learning for robotic grasping detection with knowledge transferring
  - Online continual learning for interactive instruction following agents
  - A lifelong learning approach to mobile robot navigation
    
- **Architectural Approaches**: Parameter-efficient adapters, progressive networks.
  - Lossless adaptation of pretrained vision models for robotic manipulation
  - Spawnnet: Learning generalizable visuomotor skills from pre-trained network
  - Minimally invasive morphology adaptation via parameter efﬁcient ﬁne-tuning
  - Efﬁcient policy adaptation with contrastive prompt ensemble for embodied agents
  - Sim-to-real robot learning from pixels with progressive nets
  
- **Replay Approaches**: Episodic memory, adversarial replay, generative replay.
  - Online object model reconstruction and reuse for lifelong improvement of robot manipulation
  - Task-unaware lifelong robot learning with retrieval-based weighted local adaptation
  - Continual learning for anthropomorphic hand grasping
  - Graspagent 1.0: Adversarial continual dexterous grasp learning
  - Lotus: Continual imitation learning for robot manipulation through unsupervised skill discovery
  - Embodied lifelong learning for task and motion planning

### In-context Knowledge Consolidation
- **Map-based Knowledge**:
  - Think global, act local: Dual-scale graph transformer for vision-and-language navigation
  - Instruction-guided path planning with 3d semantic maps for vision-language navigation
  - A persistent spatial semantic representation for high-level natural language instruction execution
  - Film: Following instructions in language with modular methods
  - Poni: Potential functions for objectgoal navigation with interaction-free learning
  - Tag map: A text-based map for spatial reasoning and navigation with large language models
  - 
- **Scene Graphs**: Efficient querying and environment structuring.
  - 
- **Skill Libraries**:
  - 
- **Log Data**: Storage for temporal event data and rapid environment updates.

## Knowledge Refinement
### Space and Time Efficiency
- **Knowledge Distillation**: DEDER (Distilling Embodied-Relevant Knowledge from LLMs for Interactive Embodied Navigation Task).
- **Quantization**: Quantization-aware training for efficient deployment.
- **Pruning**: Structured and unstructured pruning techniques for embodied AI models.

### Generalization
- Methods for improving model generalization through data efficiency, efficient compression, and balanced knowledge retention.


