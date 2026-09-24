import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicAgentAllocator(nn.Module):
    """
    Allocateur dynamique d'agents spécialisés.
    Chaque agent a une architecture et des poids différents, adaptés à un type de tâche.
    """
    
    def __init__(self, emb_dim: int, num_agents: int = 4, lr_online: float = 0.05):
        """
        Args:
            emb_dim: Dimension des embeddings
            num_agents: Nombre d'agents spécialisés
            lr_online: Taux d'apprentissage en ligne pour adaptation des poids
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.num_agents = num_agents
        self.lr_online = lr_online
        
        # Sélecteur d'agents
        self.agent_selector = nn.Linear(emb_dim, num_agents)
        
        # Priors d'agents (apprises et adaptables)
        self.register_buffer("agent_priors", torch.ones(num_agents) / num_agents)
        
        # Quatre agents spécialisés
        self.agents = nn.ModuleList([
            # Agent 1 : Reasoning
            nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            # Agent 2 : Math
            nn.Sequential(
                nn.Linear(emb_dim, emb_dim * 2),
                nn.GELU(),
                nn.Linear(emb_dim * 2, emb_dim),
                nn.Dropout(0.1)
            ),
            # Agent 3 : Logic
            nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.Tanh(),
                nn.Dropout(0.1)
            ),
            # Agent 4 : Critic
            nn.Sequential(
                nn.Linear(emb_dim, emb_dim // 2),
                nn.ReLU(),
                nn.Linear(emb_dim // 2, emb_dim),
                nn.Dropout(0.1)
            )
        ])
    
    def forward(self, context_emb: torch.Tensor) -> torch.Tensor:
        """
        Calcule les poids d'allocation pour chaque agent.
        
        Args:
            context_emb: (batch_size, emb_dim) - Embedding du contexte
        
        Returns:
            weights: (batch_size, num_agents) - Poids normalisés pour chaque agent
        """
        # Logits du sélecteur + priors
        logits = self.agent_selector(context_emb) + torch.log(self.agent_priors + 1e-8)
        
        # Normalisation softmax
        weights = F.softmax(logits, dim=-1)
        
        return weights
    
    def get_agent_outputs(self, context_emb: torch.Tensor) -> dict:
        """
        Génère les sorties de tous les agents.
        
        Args:
            context_emb: (batch_size, emb_dim)
        
        Returns:
            dict avec outputs de chaque agent
        """
        agent_names = ["reasoning", "math", "logic", "critic"]
        outputs = {}
        
        for i, (agent, name) in enumerate(zip(self.agents, agent_names)):
            outputs[name] = agent(context_emb)
        
        return outputs
    
    def update_continuous_weights(self, agent_losses: torch.Tensor):
        """
        Adaptation en ligne des poids d'agents basée sur leurs performances.
        
        Args:
            agent_losses: (num_agents, ) - Perte pour chaque agent
        """
        with torch.no_grad():
            # Gradient d'amélioration
            gradient = -agent_losses  # Meilleur agent = gradient positif
            
            # Mise à jour addititive + normalisation
            updated_log_priors = torch.log(self.agent_priors + 1e-8) + self.lr_online * gradient
            self.agent_priors = F.softmax(updated_log_priors, dim=-1)
