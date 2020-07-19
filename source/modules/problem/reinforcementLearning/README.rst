**********************
Reinforcement Learning
**********************

Describes a problem were we want to solve a sequential decision making problem. Here we are given an environment such that for an action :math:`a` a state :math:`s` is mapped to state :math:`s'` while giving a reward :math:`r` with probability :math:`p(s',r|s,a)`. The goal is to find a policy :math:`\pi` that choses an action :math:`a` for a given state :math:`s` with probability :math:`\pi(a|s)` such that for every state :math:`s` the chosen action :math:`a` is such that the value function

.. math::

    V^\pi(s)=\E_{\substack{a_t\sim \pi(\cdot|s_t) \\ s_{t+1},r_t\sim p(\cdot,\cdot|s_t,a_t)}}\left[\sum\limits_{t=0}^\infty \gamma^t r_t\bigg|s_0=s\right]

is maximal. Here :math:`\gamma` is the discount factor.