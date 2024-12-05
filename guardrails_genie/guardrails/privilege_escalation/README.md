# Privilege Escalation

## Overview

Privilege escalation is the process of gaining more access to resources/machines/softwares/systems than one should have. There are many categorizations of privilege escalation.

* Horizontal Privilege Escalation: Gaining access to resources that are on the same level of access as one's current access but on a different resource/machine/software/system.
* Vertical Privilege Escalation: Gaining access to resources on the same resource/machine/software/system that are on a higher level of access than one's current access.

This can be achieved by exploiting vulnerabilities in the system. The most common way to exploit vulnerabilities is through **exploitation**. Exploitation is the process of finding and exploiting vulnerabilities in a system to gain access to resources that one should not have access to. Usually this is an iterative process of finding vulnerabilities and exploiting them.

Any LLM based system will have levels of abstractions. Most such applications will have the "conventional" application layer with security baked in. What we are interested is in the "intelligence" layer where the logic is executed/reasoned about using LLMs. 

At this layer, the input is a string which remains harmless to the "application" layer and thus bypass the "conventional" security mechanisms.

Imagine a system where the "intelligence" layer has access to Python REPL. This REPL is not exposed to the user interface and is used by the LLM to execute code if required (tool call). The attack prompt can access this REPL and execute any code with the same privileges as the LLM process. This escalated privilege is something we want to avoid using "Privilege Escalation Guardrails".

Privilege Escalation is part of the "Jailbreak" category of attacks. The paper [Jailbreaking ChatGPT via Prompt Engineering: An Empirical Study](https://arxiv.org/pdf/2305.13860v2) categorizes Jailbreaks into three top level categories:

![image](../../../docs/assets/priv_esc_taxonomy.png)

Note that the taxonomy is built by analyzing the prompt engineering attacks on ChatGPT. This may or maynot transfer to other systems but is a good starting point to narrow down the strings/text that we need to guard against. Privilege Escalation is somewhat dependent on the way the system is designed. Given the gold rush to automate parts of a product/business/process, it may not be farfetched to say that we will see all kinds of privilege escalation attacks in the future.

## Definition

As per the paper, Privilege Escalation is a distinct category of prompts that seek to directly circumvent the imposed restrictions. In contrast to the previous categories, these prompts attempt to induce the model to break any of the restrictions in place, rather than bypassing them. Once the attackers have elevated their privilege level, they can ask the prohibited question and obtain the answer without further impediment.

## Guardrails

This section outlines our approach to defend against privilege escalation. This is not in any way covering all bases but is a stab at understanding the problem space and designing simple systems to avoid privilege escalation.

