# Paper List of Knowledge Evolution For Lifelong Embodied AI

We introduce the **Knowledge Evolution for Lifelong Embodied AI** to help position existing research within this ultimate goal and identify key gaps that remain. We welcome your valuable feedback and suggestions to enhance the discussion on lifelong embodied AI. If you have ideas to share, please feel free to raise issues, submit pull requests, or reach out to us directly at zekewang@outlook.com.

Thank you for your support and collaboration!


## Table of Contents
- [0. Introduction](#0-introduction)
- [1. Embodied Agent Initialization](#1-embodied-agent-initialization)
- [2. Data Collection](#2-data-collection)
  - [Human-Curated Data](#human-curated-data)
  - [Active Data Collection](#active-data-collection)
- [3. Knowledge Consolidation](#3-knowledge-consolidation)
  - [In-parameter Knowledge Consolidation](#in-parameter-knowledge-consolidation)
  - [In-context Knowledge Consolidation](#in-context-knowledge-consolidation)
- [4. Knowledge Refinement](#4-knowledge-refinement)
  - [Space and Time Efficiency](#space-and-time-efficiency)
  - [Generalization](#generalization)

## 0 Introduction
Lifelong embodied AI aims to enable autonomous agents to evolve continuously by acquiring, consolidating, and refining knowledge over time. This repository follows a structured approach to represent the major components of lifelong embodied AI.

## 1 Embodied Agent Initialization
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

## 2 Data Collection
### Human-Curated Data
A breif collection of Embodied AI tasks, more sophisticated can refer to [Embodied AI Paper List](https://github.com/HCPLab-SYSU/Embodied_AI_Paper_List).
- **Robotic Manipulation**:
  - Jacquard: A Large Scale Dataset for Robotic Grasp Detection
  - ACRONYM: A Large-Scale Grasp Dataset Based on Simulation
  - MultigripperGrasp: A Dataset for Robotic Grasping from Parallel Jaw Grippers to Dexterous Hands
  - Benchmark for human-to-robot handovers of unseen containers with unknown ﬁlling
  - Robust robotic pouring using audition and haptics
  - Rearrangement: A challenge for embodied ai
  - Lohoravens: A long-horizon language-conditioned benchmark for robotic tabletop manipulation
  - Cliport: What and where pathways for robotic manipulation
  - Robonet: Large-scale multi-robot learning
  - Rlbench: The robot learning benchmark & learning environment
  - Bc-z: Zero-shot task generalization with robotic imitation learning
  - Open XEmbodiment: Robotic learning datasets and RT-X models
  - Droid: A large-scale in-thewild robot manipulation dataset
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
  - Embodied question answering
  - Iqa: Visual question answering in interactive environments
  - S-eqa: Tackling situational queries in embodied question answering

### Active Data Collection

- **Low-level Perceptual Curiosity**:
  > An insightful survey you can refer to is *[Interactive Perception: Leveraging Action in Perception and Perception in Action](https://arxiv.org/abs/1604.03670)*
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

## 3 Knowledge Consolidation
### In-parameter Knowledge Consolidation
> For an in-depth dive into continual learning, you can check *[A Comprehensive Survey of Continual Learning: Theory, Method and Application](https://ieeexplore.ieee.org/abstract/document/10444954?casa_token=rtsIfvzxseUAAAAA:lWUT6qxLxOk1BRwrokexAiMSM7wGhnI5QWErNixMzoqp-sfjmNWlxEAF8jF5wG4pkOqVWAOPWg)*
- **Regularization Approaches**:
  > Some classic approaches include
  >  - Learning without Forgetting [[IEEE](https://ieeexplore.ieee.org/abstract/document/8107520?casa_token=eKanaS0lU2YAAAAA:G3VXEK27gn7X8JmSkYMABjPTLBqFSyOPD--hYG0BZGhMcf98-awrv0XBLW5VBD3ABgvbbVTvkg)]
  >  - Overcoming catastrophic forgetting in neural networks [[PNAS](https://www.pnas.org/doi/abs/10.1073/pnas.1611835114)]
  >  - Continual Learning Through Synaptic Intelligence [[PMLR](https://proceedings.mlr.press/v70/zenke17a)]
  >  - Gradient Episodic Memory for Continual Learning [[NeurIPS](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html)]
  - Language models meet world models: Embodied experiences enhance language models
  - Airloop: Lifelong loop closure detection
  - Continual learning for robotic grasping detection with knowledge transferring
  - Online continual learning for interactive instruction following agents
  - A lifelong learning approach to mobile robot navigation
    
- **Architectural Approaches**: Parameter-efficient adapters, progressive networks.
  > Some classic approaches include
  >  - Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights [[ECCV](https://openaccess.thecvf.com/content_ECCV_2018/html/Arun_Mallya_Piggyback_Adapting_a_ECCV_2018_paper.html)]
  >  - PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning [[CVPR](https://openaccess.thecvf.com/content_cvpr_2018/html/Mallya_PackNet_Adding_Multiple_CVPR_2018_paper.html)]
  >  - Expert gate: Lifelong learning with a network of experts [[CVPR](https://openaccess.thecvf.com/content_cvpr_2017/html/Aljundi_Expert_Gate_Lifelong_CVPR_2017_paper.html)]
  >  - Progressive Neural Networks [[NeurIPS](https://openreview.net/forum?id=Hy1e7Z05)]
  >  - Coscl: Cooperation of small continual learners is stronger than a big one [[ECCV](https://link.springer.com/chapter/10.1007/978-3-031-19809-0_15)]
  - Lossless adaptation of pretrained vision models for robotic manipulation
  - Spawnnet: Learning generalizable visuomotor skills from pre-trained network
  - Minimally invasive morphology adaptation via parameter efﬁcient ﬁne-tuning
  - Efﬁcient policy adaptation with contrastive prompt ensemble for embodied agents
  - Sim-to-real robot learning from pixels with progressive nets
  
- **Replay Approaches**: Episodic memory, adversarial replay, generative replay.
  > Some classic approaches include
  >  - iCaRL: Incremental Classifier and Representation Learning [[CVPR](https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html)]
  >  - Dark Experience for General Continual Learning: a Strong, Simple Baseline [[NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)]
  >  - Continual Learning with Deep Generative Replay [[NeurIPS](https://proceedings.neurips.cc/paper/2017/hash/0efbe98067c6c73dba1250d2beaa81f9-Abstract.html)]
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
  - Distilled feature ﬁelds enable few-shot language-guided manipulation
  - Clip-ﬁelds: Weakly supervised semantic ﬁelds for robotic memory
    
- **Scene Graphs**: Efficient querying and environment structuring.
  - Hydra: A realtime spatial perception engine for 3d scene graph construction and optimization
  - 3d dynamic scene graphs: Actionable spatial perception with places, objects, and humans
  - Building dynamic knowledge graphs from text-based games
  - Grid: Scene-graph-based instruction-driven robotic task planning
  - Saynav: Grounding large language models for dynamic planning to navigation in new environments
  - Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning
  - Towards coarse-grained visual language navigation task planning enhanced by event knowledge graph
    
- **Skill Libraries**:
  - Text2motion: From natural language instructions to feasible plans
  - Code as policies: Language model programs for embodied control
  - Ok-robot: What really matters in integrating open-knowledge models for robotics
  - Tulip agent–enabling llm-based agents to solve tasks using large tool libraries
  - Voyager: An open-ended embodied agent with large language models
  - Lifelong robot library learning: Bootstrapping composable and generalizable skills for embodied control with language models
  - Bootstrap your own skills: Learning to solve new tasks with large language model guidance
  - League++: Empowering continual robot learning through guided skill acquisition with large language models
    
- **Log Data**: 
  - Remembr: Building and reasoning over long-horizon spatio-temporal memory for robot navigation
  - Episodic memory verbalization using hierarchical representations of lifelong robot experience

## 4 Knowledge Refinement
### Space and Time Efficiency
- <ins>**In-parameter compression**</ins>
  - **Knowledge Distillation**:
    - Distilling internet-scale vision-language models into embodied agents
    - Embodied cot distillation from llm to off-the-shelf agents
    - Kickstarting deep reinforcement learning
    - Gridtopix: Training embodied agents with minimal supervision
    - MAGIC: Meta-Ability Guided Interactive Chain-of-Distillation for Effective-and-Efficient Vision-and-Language Navigation
    - Do We Really Need a Complex Agent System? Distill Embodied Agent into a Single Model
  
  - **Quantization**:
    - Quantization-aware imitation-learning for resource-efﬁcient robotic control
      
  - **Pruning**:
    - no papers related to embodied ai, but can refer to LLM related pruning strategies.
      
- <ins>**In-context compression**</ins>
  - Database compression technique (text):
    - [InnoDB](https://dev.mysql.com/doc/refman/8.0/en/innodb-introduction.html)
  - Spatial data compression
    - [OctoMap](https://octomap.github.io/)
    - HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression
    - ContextGS: Compact 3D Gaussian Splatting with Anchor Level Context Model
    - Compact3D: Compressing Gaussian Splat Radiance Field Models with Vector Quantization

### Generalization
> Unlike training-stage generalization, refinement-stage generalization focuses on preserving performance when reducing model complexity as first discussed in "Optimal brain damage (1989)"
- Embodied cot distillation from llm to off-the-shelf agents
- Quantization-aware imitation-learning for resource-efﬁcient robotic control
- MAGIC: Meta-Ability Guided Interactive Chain-of-Distillation for Effective-and-Efficient Vision-and-Language Navigation
  
> Training-stage generalization papers for reference:
> - Data augmentation: Reinforcement learning with augmented data
> - Model Design: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
> - RLHF Training strategy: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning (not embodied AI but provides evidence)


