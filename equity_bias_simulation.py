\documentclass[lettersize,journal]{IEEEtran}
\usepackage{amsmath,amsfonts}
%\usepackage{algorithmic}
%\usepackage{algorithm}
\usepackage{array}
\usepackage[caption=false,font=normalsize,labelfont=sf,textfont=sf]{subfig}
\usepackage{textcomp}
\usepackage{stfloats}
\usepackage{url}
\usepackage{verbatim}
\usepackage{graphicx}
\usepackage{cite}
\usepackage{multirow}
\usepackage{url}
\usepackage{footnote}
\usepackage{hyperref}

\usepackage{tabularray}

\usepackage{algorithm}
\usepackage{algpseudocode}


\usepackage{array}
\usepackage{booktabs}


\usepackage{color}
\usepackage{xcolor}

\hyphenation{op-tical net-works semi-conduc-tor IEEE-Xplore}
% updated with editorial comments 8/9/2021
 
\begin{document}
 
\title{Industrial Control Systems Attack Surface: From Theory to Practice / Autonomous Defense}

%\author{Zahra Dehghani Mohammadabadi, Abbas Yazdinejad, Ali Dehghantanha,~\IEEEmembership{Senior Member,~IEEE }, Gautam Srivastava,~\IEEEmembership{Senior Member,~IEEE }
        % <-this % stops a space 
%\thanks{Z. Dehghani Mohammadabadi is with the School of Computer Engineering, Iran University of Science and Technology, Tehran, Iran. email: dehghani.m.zahra@gmail.com}% <-this % stops a space
%\thanks{A. Yazdinejad and A. Dehghantanha are with the Cyber Science Lab, Canada Cyber Foundry, University of Guelph, ON, Canada, email: ayazdine@uoguelph.ca, adehghan@uoguelph.ca}
%\thanks{G. Srivastava is with the Department of Mathematics and Computer Science, Brandon University, Brandon R7A 6A9, Canada, and also with the Research Centre of Interneural Computing, China Medical University, Taichung 40402, Taiwan as well as with the Department of Computer Science and Math, Lebanese American University, Beirut 1102, Lebanon (e-mail: srivastavag@brandonu.ca).}}

% The paper headers
\markboth{IEEE TBD}%
{Shell \MakeLowercase{\textit{et al.}}: A Sample Article Using IEEEtran.cls for IEEE Journals}

%\IEEEpubid{0000--0000/00\$00.00~\copyright~2021 IEEE}
% Remember, if you use this you must call \IEEEpubidadjcol in the second
% column for its text to clear the IEEEpubid mark.

\maketitle

\begin{abstract}
Industrial Control Systems (ICS) are vital for critical infrastructure, governing operations in sectors like power plants and water treatment facilities. However, these systems are vulnerable to attacks, especially through manipulations of sensor or actuator values, posing significant security challenges. A promising technique
for detecting such attacks is machine-learning-based anomaly
detection, but it does not identify the effect of the attacked sensor or actuator
or physical anomaly’s root cause in the security of ICS operators. This paper addresses the need to identify and understand the attack surface within ICS cybersecurity, focusing on the properties of attacks that affect the security of ICS environments. In this paper, we first formally define the concept of ICS attack surface as a security metric, establishing a foundational framework for future research in this domain. Next, we introduce novel attack surface identification methods tailored for ICS, offering innovative solutions to address existing gaps in the field. Thirdly, we provide actionable recommendations for researchers and practitioners when designing and deploying attack surface identification methods for physical attacks in ICS, facilitating informed decision-making in cybersecurity strategies. Finally, we rigorously evaluate these proposed methods to compare their accuracy in identifying the impact of manipulated features in ICS networks, contributing valuable insights into the effectiveness of different approaches.
By making these contributions, our work advances the understanding and defense capabilities of ICSs against emerging cyber threats, paving the way for more robust cybersecurity strategies in critical infrastructure.
\end{abstract}



\begin{IEEEkeywords}
ICS, Attack Surface, Physical Interface Components, Operational Technology.
\end{IEEEkeywords}

\section{Introduction}
Industrial control systems (ICS) govern our critical infrastructure, including power grids, water treatment, and manufacturing processes. These systems are fundamental to the safety and functionality of such sectors, yet they are prime targets for attackers aiming to infiltrate ICS and interfere with their physical processes. Prominent examples of such attacks include Stuxnet \cite{sun}, attacks on the Ukrainian power grid \cite{uk}, and an attack on a German blast furnace \cite{gen}. 

ICS operates on process-level data: data from sensors, which read information from a physical process, and data from actuators, which send commands to control a physical process. A commonly studied type of attack involves manipulating process-level data of an ICS: an attacker gains access to an ICS and manipulates one or more sensor or actuator values, causing the ICS to react in a harmful way, such as causing a tank to overflow or a reactor to overheat.

These vulnerabilities highlight the critical need for robust security measures in ICS. ICSs are the backbone of such infrastructure, governing operations in sectors such as power plants and water treatment facilities. However, the susceptibility of these systems to cyber attacks, particularly through manipulations of sensor or actuator values, can lead to significant physical harm. While machine learning (ML)-based anomaly detection has emerged as a promising technique for identifying such attacks, it often lacks the ability to pinpoint which specific sensor or actuator was manipulated to trigger the anomaly. This limitation poses challenges for ICS operators in diagnosing the root cause of anomalies.

To address this, attribution methods aim to identify the features or components that caused an anomaly detection model to raise an alarm. These methods provide insights into the potential causes of anomalies by attributing them to specific system elements or actions. However, despite their potential, there remains a gap in the practical application of attribution methods within the context of ICS security. In high-dimensional data environments typical of ICS, attribution methods can struggle to accurately identify causal relationships due to the sheer number of interdependent variables. Also, attribution methods are limited to applying explainability tools such as SHAP, LEMNA and LIME. Additionally, sophisticated attackers can craft attacks that specifically evade detection by masking their operations as normal variations, thereby fooling both anomaly detection and attribution methods. It remains unclear how well these methods perform in accurately identifying the impact of manipulated features within an ICS network, especially in scenarios involving physical attacks.

Given these challenges, bridging the gap between anomaly detection and attribution methods in ICS security necessitates a comprehensive understanding of the attack surface. The attack surface includes all the possible points where an unauthorized user can try to enter data to or extract data from an environment. By focusing on the properties of ICS attacks that influence the accuracy of these methods—such as the timing of anomaly detection, the extent of manipulation, and the specific components targeted (sensors and actuators)—attack surface identification methods can significantly enhance the effectiveness of existing security measures. Addressing these factors enables a more robust and proactive defense mechanism, substantially improving the resilience of ICS against the evolving landscape of cyber threats.

In this work, we investigate two research questions:

\begin{itemize}
    \item \textbf{RQ1:} What properties of ICS attacks affect the accuracy of attack surface identification in ICS? For example, does the timing of anomaly detection, the magnitude of manipulation, or the type and number of components attacked (sensors and actuators) influence accuracy?
    
    \item \textbf{RQ2: }Can the attack surface identification method accurately identify the impact of manipulated features in physical components of ICS environments? %Which method is the most accurate?
\end{itemize}

To investigate these research questions, we conduct a comprehensive comparative evaluation of physical attacks in ICS. Our evaluation spans a diverse array of scenarios, encompassing two datasets from prior research that contain documented real-world ICS attacks.  Following this, we introduce an innovative attack surface method in ICS aimed at presenting the effect of identifying and assessing manipulated features of physical components in an ICS environment. This method is rigorously tested across ICS datasets to determine its effectiveness in pinpointing the specific properties of attacks—such as the timing of anomaly detection and the scale of manipulation—that critically influence the robustness of ICS security measures. We make the following contributions:

\begin{itemize}
    \item This is the first work to formally define the ICS attack surface concept as a security metric. We also introduced an attack surface measurement in ICS and how we can mitigate an ICS’s security risk by reducing the ICS’s attack surface.
    %\item We introduce novel attack surface identification methods for ICS.
    \item We provide provide actionable recommendations for the design and deployment of attack surface identification methods, this research aids both researchers and practitioners in enhancing the security of physical components in ICS environments.

     \item  Through the systematic analysis of how different attack properties affect identification, this work deepens the understanding of attack dynamics within ICS, thereby aiding in the development of more robust defense mechanisms.
    
    \item Utilizing real-world data and scenarios to demonstrate the application of the proposed attack surface ensures that the research is grounded in practical, operational realities of ICS security, making it directly applicable and beneficial for industry practitioners.
\end{itemize}

\section{Background and Related Work}
In this section, we discuss the background from:
ICS attacks and datasets (Sec. II-A), Attacks surface concept (Sec. II-B), and ICS anomaly-detection and attribution methods (Sec. II-C). 
 
\subsection{Attacks Surface}
The concept of an attack surface in cybersecurity commonly refers to the sum of different points or \textit{attack vectors} in a software environment through which unauthorized access can be gained or malicious code can be executed. These vectors typically include the software's entry and exit points, the communication channels it uses, and the data items it handles. This concept has been fundamental in understanding and securing software systems \cite{ghavamnia2020temporal}.
In addition to software security, the concept of the attack surface has been applied to various domains in network security, such as cloud security, mobile device security, automotive security, and Moving Target Defense (MTD) \cite{zhang2018network}. This extension redefines the "network attack surface" as a metric for evaluating a network's resilience against potential attacks, including zero-day attacks. It encompasses the aggregate of potential vulnerabilities in a network's configuration, considering both known and unknown threats. This expanded view considers the interconnected nature of network resources and the pathways an attacker might exploit, offering a more dynamic assessment of a network's security posture \cite{zhang2018network}.
Given the critical nature of ICS, it is imperative to establish a precise definition of the ICS attack surface. We address this crucial requirement in Section A, outlining the specific aspects and vulnerabilities unique to ICS environments, thereby enhancing our strategic approach to mitigating potential security threats.
\subsection{ICS attacks and datasets}
ICS are integral networks that manage and monitor industrial processes. These systems gather critical data through sensors and implement process control via actuator commands. However, the security of ICS is a significant concern as unauthorized access can lead to manipulation of sensors and actuators, resulting in improper system responses and potentially catastrophic outcomes.
Attackers, when gaining access to an ICS, may stealthily alter the data flowing through the system. Such manipulations often involve false-data injection attacks that cleverly evade conventional ICS security measures like state estimation and programmable logic controllers. To model these threats, we adopt an established attacker model where an adversary strategically alters sensor or actuator outputs over time to disguise their malicious activities. The impact of these alterations is profound. For instance, misleading sensor readings, such as those indicating incorrect water levels in a tank, can prompt ICS controllers to initiate inappropriate actions, like overfilling a tank, thus fulfilling the attacker's destructive goals. Our analysis utilizes two publicly available datasets from water treatment facilities, which document numerous instances of ICS manipulations. These datasets detail the specific sensors and actuators targeted, describe the methods of manipulation, and outline the attackers' objectives. By examining these scenarios, we aim to underscore the realistic and potentially severe consequences of such cyber-attacks, validating the need for robust ICS security strategies.

\subsection{Anomaly-Detection and Attribution Methods}

In the domain of ICS, the detection of anomalies and the attribution of their causes are critical for maintaining system integrity. Anomaly detection utilizes both statistical methods and deep-learning approaches to monitor and analyze real-time process values effectively.

Statistical methods, such as the PASAD algorithm, employ a departure score to identify anomalies by projecting new input data onto a lower-dimensional signal subspace derived during the training phase. Distances from the subspace centroid are measured, and scores exceeding a set threshold signal an anomaly \cite{passd}. Similarly, auto-regressive models forecast future process values from historical data, using the cumulative sum of prediction errors to detect deviations from expected patterns, providing a robust basis for anomaly detection in ICS \cite{sundararajan2017axiomatic}.

Advancing beyond traditional statistical methods, deep learning offers enhanced capabilities through models like convolutional neural networks (CNNs), gated recurrent units (GRUs), and long short-term memory networks (LSTMs) \cite{zizzo2020adversarial}. These models are particularly effective in unsupervised settings, where labeled data is scarce. CNNs, for instance, utilize one-dimensional convolutional kernels to capture temporal patterns across time series data, making them suitable for complex ICS environments \cite{apruzzese2023role}. Both GRUs and LSTMs incorporate mechanisms to manage and remember information over extended periods, which is crucial for identifying subtle and complex patterns indicative of potential security breaches \cite{gm}.

Attribution methods complement anomaly detection by pinpointing the specific features or inputs that significantly influence the detection outcomes. Employing both black-box and white-box approaches, these methods enhance the interpretability of anomaly detection models. Black-box methods, such as LIME and SHAP, estimate the impact of each input feature on model predictions through perturbations and subsequent local model approximations \cite{erion2021improving, tang2023gru}. White-box methods delve deeper, utilizing gradients within the model's architecture to assess how changes in input values alter predictions, thereby improving the accuracy and reliability of the attribution \cite{tang2023gru, smilkov2017smoothgrad}.

Together, these anomaly detection and attribution methods form a comprehensive approach for safeguarding ICS against a variety of cyber threats, ensuring that anomalies are not only detected but also thoroughly analyzed to understand their causative factors.

\section{Problem Statement}
As the complexity and connectivity of Industrial Control Systems (ICS) increase, their attack surface also expands. This attack surface encompasses all potential points through which unauthorized access could be attempted, making the system increasingly vulnerable to attacks. Therefore, it is crucial to develop a methodological approach to quantify and subsequently reduce the attack surface. This approach primarily aims to minimize the exposure of physical components—sensors, actuators, valves, meters, and control units—that are crucial to the operation of ICS and represent potential cyber-physical threat points.

The reduction of the attack surface begins by identifying all physical components that contribute to it, denoted as \( P \). These include sensors, actuators, valves, meters, and control units integral to the system. The attack surface, \( \mathcal{A} \), of the ICS can then be quantified using the following formula:
\begin{equation}
\mathcal{A}(P, T) = \{(p, t) \mid p \in P, t \in T_p\}
\label{eq:attack_surface}
\end{equation}
where \( P \) represents the set of physical components and \( T \) denotes the types of attacks, with \( T_p \) as the subset of attack types applicable to each component \( p \).

Following this, the evaluation of the exposure (\( E(p) \)) and impact (\( I(p) \)) of each component contributes to the calculation of the attack surface metric (\( M \)), which is determined by:
\begin{equation}
M(p) = E(p) \times I(p) \times ATS(p)
\label{eq:metric}
\end{equation}
where \( ATS(p) \) is the aggregate attack type severity for each component \( p \), calculated as follows:
\begin{equation}
ATS(p) = \sum_{t \in T_p} \text{Value}(t)
\label{eq:severity}
\end{equation}

Finally, strategies are implemented to minimize the metrics (\( M(p) \)) for the most critical components, thereby reducing the overall attack surface. The aim is to lower the summation of all component metrics through strategic security enhancements and design modifications, quantitatively expressed as:
\begin{equation}
\text{Total Attack Surface} = \sum_{p \in P} M(p)
\label{eq:total_surface}
\end{equation}

This methodological framework ensures a proactive approach to securing ICS by quantifying and minimizing vulnerabilities, thereby enhancing the resilience of the system against potential cyber threats.




\textbf{Integration with Systemic Security Management:} The attack surface measurement serves not only as a metric of current security posture but also guides the prioritization of mitigation strategies, helping administrators focus on areas with the highest risk potential. This systematic approach to quantifying and reducing the attack surface is vital for maintaining the integrity and security of ICS in the face of evolving cyber threats. By systematically reducing the attack surface, organizations can decrease the number of potential attack vectors, making it significantly more difficult for attackers to gain unauthorized access and manipulate the system. This focused reduction not only enhances the security posture of the ICS but also aligns with proactive risk management practices that are essential for the protection of critical infrastructure.


%\subsection{Threat Model}
%We systematically
%synthesize a set of possible external attack vectors as
%a function of the attacker’s ability to deliver malicious
%input via particular modalities: indirect physical access,
%short-range wireless access, and long-range wireless
%access. Within each of these categories, we characterize
%%the attack surface exposed in current automobiles and
%their surprisingly large set of I/O channels.

%operational capabilities characterize the
%adversary’s requirements in delivering a malicious input
%to a particular access vector in the field
\subsection{Threat Model}
Our threat model systematically synthesizes possible external attack vectors as a function of the adversary's capability to deliver malicious input via various modalities: indirect physical access, short-range wireless access, and long-range wireless access. Each modality represents a distinct method that could impact the physical components of an Industrial Control System (ICS), such as sensors, actuators, and valves. The significance of these vectors is amplified by the diverse and numerous I/O channels through which modern ICS are accessed and controlled, thereby broadening the scope for potential exploitation.

\textbf{Operational Capabilities Required by the Adversary:}
To effectively exploit these modalities, adversaries must possess a set of specialized capabilities. Technical knowledge (\(C_{\text{tech}}\)), which includes understanding and manipulating ICS operations and familiarity with both software and hardware configurations, is crucial. Additionally, access to technology (\(C_{\text{access}}\)) that can interface with ICS components at various ranges, including specialized equipment for intercepting or emitting signals, is required. Moreover, the ability to bypass or deceive existing protective measures such as encryption, authentication protocols, and physical security barriers (\(C_{\text{security}}\)) is essential for a successful attack.

\textbf{Case Study and Attack Examples:}
Through real-world data from documented scenarios in the SWaT and WADI datasets, our threat model's application is illustrated. These examples demonstrate a broad range of attack vectors along with the operational capabilities required to execute them effectively.

\textbf{Example Attack Scenarios:}
In one scenario, an attack on Sensor \(S\) at location \( \text{MV101} \) through indirect physical access (\(I_{\text{indirect}}\)) resulted in a tank overflow between "28-12-2015 10:29:14" and "10:44:53". This attack disrupted system stability and could lead to environmental hazards or safety incidents. The capabilities required included \(C_{\text{tech}}\) and \(C_{\text{access}}\) to manipulate sensor readings without direct physical interaction. Another scenario involved an attack via short-range wireless access (\(I_{\text{short}}\)) targeting the control unit \(C\) at \( \text{P602} \), which induced a system freeze from "30-12-2015 01:42:34" to "01:54:10". This attack caused operational downtime and potential cascading effects throughout the facility. The necessary capabilities for this attack were \(C_{\text{tech}}\) and \(C_{\text{security}}\), including the ability to bypass digital security measures to freeze system operations.


These examples align with the defined threat model, illustrating how attackers leverage different modalities and capabilities to exploit the attack surface of an ICS. By analyzing these practical scenarios, we can better understand the dynamic interplay of attack vectors, target vulnerabilities, and the attacker's operational capabilities, guiding more effective defense strategies. This comprehensive analysis helps in prioritizing security enhancements, such as strengthening wireless security protocols and improving physical access controls.




\section{The Industrial Control Systems Attack Surface Model}
When we focus on ICS, the attack surface concept takes on additional layers of complexity. The attack surface for ICS includes all potential points through which an unauthorized user might access, manipulate, or disrupt the system's operations. Notably, this includes physical interface components such as sensors and actuators, which are crucial in ICS environments. The ICS attack surface uniquely covers operational technology (OT) aspects, encompassing the systems that directly control or monitor physical processes.

%\textbf{Definition:} \textit{The attack surface of an ICS comprises all the potential physical points through which unauthorized access, manipulation, or disruption to the system's operations might occur. This specifically includes the system's physical interface components, such as sensors and actuators, which are integral in field-level operations. Distinctively, the ICS attack surface encapsulates Operational Technology aspects, emphasizing the critical role of physical processes that are directly controlled or monitored by the system.} 

\textbf{Definition:} \textit{The attack surface of an ICS comprises all the potential physical points through which unauthorized access, manipulation, or disruption to the system's operations might occur. This includes:
\textbf{1- Physical Interface Components:} Such as sensors and actuators, which are integral in field-level operations and serve as primary physical entry points to the system. \textbf{2- Communication Channels:} The data conduits between system components, including control units and external networks, which are crucial for the system's operation and are potential vectors for data manipulation or interception.
 \textbf{3- Untrusted Data Items:} Data that may be subject to tampering or spoofing, such as sensor readings or control signals, which can be manipulated to disrupt system operations.}
 
\textit{Distinctively, the ICS attack surface encapsulates OT aspects, emphasizing the critical role of physical processes that are directly controlled or monitored by the system.}









In this definition, the emphasis is solely on the physical aspects, recognizing the unique vulnerabilities and security considerations that arise from the real-world, tangible elements of ICS. It is important to emphasize that the ICS attack surface is significantly influenced by these physical interfaces, in contrast to traditional attack surfaces that are primarily associated with digital interfaces such as software systems and network connections. The focus on digital interfaces, including software entry and exit points, communication channels, and data management, has been the primary concern in the conventional understanding of attack surfaces. Similarly, when the concept is extended to network attack surfaces, the emphasis remains on digital aspects, evaluating the resilience of a network against potential attacks by assessing vulnerabilities in network configurations, including both known and unknown elements.
However, the ICS attack surface diverges from this digital-centric view by incorporating physical interfaces at the field level, which directly interact with the physical world. This distinction is crucial as ICS operates at the intersection of digital computing and physical processes. The inclusion of physical interfaces in the ICS attack surface highlights that threats to these systems can arise from both digital vulnerabilities and physical access or manipulation. This broader scope is vital for accurately assessing and mitigating risks in ICS, which have direct control over and monitor essential physical operations.

\subsection{Formal Model for ICS Attack Surface}
Let us consider a comprehensive model for the attack surface in an ICS that encompasses various physical components and considers multiple types of cyber-physical attacks. We define \( P \) as the set of all physical components within an ICS, where each element can be a sensor, actuator, valve, meter, or control unit. Represented as Equation \ref{q1}:
\begin{equation}\label{q1}
\small
P = S \cup A \cup V \cup M \cup C
\end{equation}
where \( S \) denotes sensors, \( A \) denotes actuators, \( V \) denotes valves, \( M \) denotes meters, and \( C \) denotes control units.
 
For attack types, we define \( T \) to represent the types of attacks that can target these components, with each type assigned a specific severity value:
\begin{itemize}
    \item \( t_1 = \text{Single Stage Single Point (SSSP)}, \text{Value} = 1 \) — Represents basic attacks targeting a single component at one point in time.
    \item \( t_2 = \text{Single Stage Multi-Point (SSMP)}, \text{Value} = 2 \) — Represents attacks targeting multiple components simultaneously at one point in time.
    \item \( t_3 = \text{Multi Stage Single Point (MSSP)}, \text{Value} = 3 \) — Represents attacks that occur in multiple stages affecting a single component, indicating higher complexity.
    \item \( t_4 = \text{Multi Stage Multi-Point (MSMP)}, \text{Value} = 4 \) — The most severe, representing attacks that affect multiple components over multiple stages.
\end{itemize}

The attack surface \( \mathcal{A} \) of the ICS is defined as the set of tuples representing each component and its susceptible attack types as presented in Equation \ref{q2}:
\begin{equation}\label{q2}
\small
\mathcal{A}(P, T) = \{(p, t) \mid p \in P, t \in T_p\}
\end{equation}
where \( T_p \subseteq T \) indicates the subset of attack types applicable to component \( p \).

\textbf{Definition of ICS Attack Surface:}
Given the ICS environment \( E_{ICS} \), the attack surface of the ICS is defined as the tuple \( (\mathcal{M}_{E_{ICS}}, \mathcal{C}_{E_{ICS}}, \mathcal{I}_{E_{ICS}}) \), where:
\begin{itemize}
    \item \( \mathcal{M}_{E_{ICS}} \) is the set of physical entry points (e.g., sensors, actuators) and exit points (e.g., control signal outputs).
    \item \( \mathcal{C}_{E_{ICS}} \) is the set of communication channels (e.g., network connections, wireless interfaces).
    \item \( \mathcal{I}_{E_{ICS}} \) is the set of untrusted data items (e.g., sensor readings subject to spoofing).
\end{itemize}

\textbf{Attack Surface Comparison:}
Given two similar ICS environments \( E_{ICS1} \) and \( E_{ICS2} \), the attack surface of \( E_{ICS1} \) is considered larger than that of \( E_{ICS2} \) if:
\begin{enumerate}
    \item \( \mathcal{M}_{E_{ICS1}} \supset \mathcal{M}_{E_{ICS2}} \) and \( \mathcal{C}_{E_{ICS1}} \supseteq \mathcal{C}_{E_{ICS2}} \) and \( \mathcal{I}_{E_{ICS1}} \supseteq \mathcal{I}_{E_{ICS2}} \), or
    \item \( \mathcal{M}_{E_{ICS1}} \supseteq \mathcal{M}_{E_{ICS2}} \) and \( \mathcal{C}_{E_{ICS1}} \supset \mathcal{C}_{E_{ICS2}} \) and \( \mathcal{I}_{E_{ICS1}} \supseteq \mathcal{I}_{E_{ICS2}} \), or
    \item \( \mathcal{M}_{E_{ICS1}} \supseteq \mathcal{M}_{E_{ICS2}} \) and \( \mathcal{C}_{E_{ICS1}} \supseteq \mathcal{C}_{E_{ICS2}} \) and \( \mathcal{I}_{E_{ICS1}} \supset \mathcal{I}_{E_{ICS2}} \).
\end{enumerate}

%This structured approach allows for a clear and formal comparison of attack surfaces between different ICS configurations, facilitating more effective security assessments and enhancements.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%\textbf{Total Attack Surface Calculation:}
%Aggregate the metrics for all components to compute the total ICS attack surface:
%\[
%\text{Total Attack Surface} = \sum_{p \in P} M(p)
%\]







%\subsection{Motivating Example and Assumptions}

%Consider a simple ICS with three distinct physical components, each with defined exposure and impact scores, and susceptible to different attack types:

%\begin{itemize}
 %   \item Sensor \( S \): \( E(s) = 0.8, I(s) = 0.7, T_s = \{t_1, t_2\} \)
%    \item Actuator \( A \): \( E(a) = 0.9, I(a) = 0.9, T_a = \{t_1, t_4\} \)
%    \item Valve \( V \): \( E(v) = 0.5, I(v) = 0.8, T_v = \{t_1, t_3\} \)
%\end{itemize}

%\textbf{Example Calculation:}
%Calculate \( ATS \) for each component and their respective attack surface metrics:
%\[
%ATS(s) = 1 + 2 = 3, \quad M(s) = 0.8 \times 0.7 \times 3 = 1.68
%\]
%\[
%ATS(a) = 1 + 4 = 5, \quad M(a) = 0.9 \times 0.9 \times 5 = 4.05
%\]
%\[
%ATS(v) = 1 + 3 = 4, \quad M(v) = 0.5 \times 0.8 \times 4 = 1.6
%\]
%\[
%\text{Total Attack Surface} = 1.68 + 4.05 + 1.6 = 7.33
%\]

%This example highlights how the model quantifies the risk associated with each physical component based on their exposure, impact, and the severity of applicable attack types, thereby facilitating prioritized and informed security


\subsection{Calculation Method for ICS Attack Surface}
We define a method to quantify the ICS attack surface along three key dimensions reflective of ICS-specific vulnerabilities: \textit{physical components, communication channels, }and\textit{ data integrity}. This quantitative approach is inspired by existing methods in system security that measure the attack surface as a function of resources' exposure and susceptibility to threats.

\textbf{Definition of ICS Attack Surface Measurement:}
\textit{Given the attack surface components \( \mathcal{P}_{E_{ICS}}, \mathcal{C}_{E_{ICS}}, \mathcal{D}_{E_{ICS}} \) of an ICS, where \( E_{ICS} \) denotes the ICS environment, the attack surface measurement is the triple 
\( \left( \sum_{p \in \mathcal{P}_{E_{ICS}}} derm(p), \sum_{c \in \mathcal{C}_{E_{ICS}}} derc(c), \sum_{d \in \mathcal{D}_{E_{ICS}}} derd(d) \right) \)}.

\textbf{Calculation Steps:}
The process of quantifying the attack surface involves several critical steps aimed at comprehensively assessing the vulnerability of the Industrial Control Systems (ICS). Initially, components within the ICS environment are identified, encompassing the set of physical components such as sensors and actuators (\( \mathcal{P}_{E_{ICS}} \)), communication channels (\( \mathcal{C}_{E_{ICS}} \)), and untrusted data items (\( \mathcal{D}_{E_{ICS}} \)). This identification serves as the foundation for subsequent calculations.

Following identification, the damage potential-effort ratio is estimated for each component, which evaluates the relative risk posed by each element within the system. Specifically, \( derm(p) \) is calculated for each physical component \( p \in \mathcal{P}_{E_{ICS}} \), \( derc(c) \) for each communication channel \( c \in \mathcal{C}_{E_{ICS}} \), and \( derd(d) \) for each data item \( d \in \mathcal{D}_{E_{ICS}} \). This ratio assesses the potential damage of an exploit relative to the effort required to execute it, providing a metric of vulnerability for each component. The final step involves aggregating these individual metrics to compute the overall attack surface of the ICS. This computation aggregates the damage potential-effort ratios across all identified components, channels, and data items, effectively summing up the risk associated with each to yield a comprehensive view of the system's vulnerability. The total attack surface is calculated using the following formula:
\begin{equation}\label{qe}
\small
\begin{split}
    \text{ICS Attack Surface} = \Bigg( & \sum_{p \in \mathcal{P}_{E_{ICS}}} \text{derm}(p), \\
    & \sum_{c \in \mathcal{C}_{E_{ICS}}} \text{derc}(c), \sum_{d \in \mathcal{D}_{E_{ICS}}} \text{derd}(d) \Bigg)
\end{split}
\end{equation}
These steps ensure a detailed and quantifiable analysis of the attack surface, enabling targeted strategies to mitigate risks and enhance the security posture of the ICS.



\textbf{Analogous to Risk Modeling:}
This measurement method aligns with risk estimation techniques used in traditional risk modeling, where the risk associated with an event is calculated as the product of the event's probability of occurrence and its consequence. In the context of ICS:
\begin{itemize}
    \item The probability \( p(x) \) of a successful attack using resource \( x \) is analogous to the likelihood that \( x \) has an exploitable vulnerability.
    \item The consequence is represented by the damage potential-effort ratio, indicating the payoff to an attacker for exploiting a given vulnerability.
\end{itemize}

In practice, due to the difficulty of accurately predicting software defects and estimating the likelihood of vulnerabilities, a conservative assumption that \( p(x) = 1 \) for all components, channels, and data items can be made. This assumes that every method might have a future vulnerability that has not yet been discovered.

\textbf{Practical Application:}
By systematically applying this model, ICS administrators and security analysts can more effectively prioritize security improvements based on the calculated attack surface, directing resources toward mitigating the most significant vulnerabilities identified by this quantitative analysis. Below, we present a structured algorithm that outlines the steps necessary for quantitatively measuring the attack surface of an ICS. This algorithm integrates the identification of system components, the estimation of their vulnerability (damage potential-effort ratio), and the aggregation of these estimations into a total attack surface measure.

\begin{algorithm}
\caption{Calculate ICS Attack Surface}
\begin{algorithmic}[1]
\State \textbf{Input:} ICS environment $E_{ICS}$
\State \textbf{Output:} Attack Surface Measurement $(ASM)$

\Procedure{IdentifyComponents}{$E_{ICS}$}
    \State Initialize $\mathcal{P}_{E_{ICS}} \gets \emptyset$, $\mathcal{C}_{E_{ICS}} \gets \emptyset$, $\mathcal{D}_{E_{ICS}} \gets \emptyset$
    \For{each component $comp$ in $E_{ICS}$}
        \If{$comp$ is a physical component}
            \State $\mathcal{P}_{E_{ICS}} \gets \mathcal{P}_{E_{ICS}} \cup \{comp\}$
        \ElsIf{$comp$ is a communication channel}
            \State $\mathcal{C}_{E_{ICS}} \gets \mathcal{C}_{E_{ICS}} \cup \{comp\}$
        \ElsIf{$comp$ is a data item}
            \State $\mathcal{D}_{E_{ICS}} \gets \mathcal{D}_{E_{ICS}} \cup \{comp\}$
        \EndIf
    \EndFor
    \State \textbf{return} $\mathcal{P}_{E_{ICS}}, \mathcal{C}_{E_{ICS}}, \mathcal{D}_{E_{ICS}}$
\EndProcedure

\Procedure{EstimateDamageRatios}{$\mathcal{P}, \mathcal{C}, \mathcal{D}$}
    \State Initialize $derm \gets \emptyset$, $derc \gets \emptyset$, $derd \gets \emptyset$
    \For{$p$ in $\mathcal{P}$}
        \State $derm(p) \gets$ EstimateDamagePotentialEffortRatio($p$)
    \EndFor
    \For{$c$ in $\mathcal{C}$}
        \State $derc(c) \gets$ EstimateDamagePotentialEffortRatio($c$)
    \EndFor
    \For{$d$ in $\mathcal{D}$}
        \State $derd(d) \gets$ EstimateDamagePotentialEffortRatio($d$)
    \EndFor
    \State \textbf{return} $derm, derc, derd$
\EndProcedure

\Procedure{CalculateAttackSurface}{$derm, derc, derd$}
    \State $ASM \gets (\sum_{m \in derm} derm(m),$ 
    \State $\phantom{ASM \gets (} \sum_{c \in derc} derc(c), \sum_{d \in derd} derd(d))$
    \State \textbf{return} $ASM$
\EndProcedure

\State $(\mathcal{P}, \mathcal{C}, \mathcal{D}) \gets$ \Call{IdentifyComponents}{$E_{ICS}$}
\State $(derm, derc, derd) \gets$ \Call{EstimateDamageRatios}{$\mathcal{P}, \mathcal{C}, \mathcal{D}$}
\State $ASM \gets$ \Call{CalculateAttackSurface}{$derm, derc, derd$}
\State \textbf{print} $ASM$
\end{algorithmic}
\end{algorithm}

This algorithm provides a step-by-step approach for calculating the ICS attack surface, from component identification through to the final computation of the attack surface metric. Each step involves specific procedures that ensure a comprehensive evaluation of the system's security posture.

\section{Methodology}


\subsection{Datasets used for attack surface evaluation}
\subsection{Implementing ICS Attack Surface Method}
\subsection{Autonomous Defense}
\subsection{Evaluation metric for ICS Attack Surface}





\section{RESULTS: EVALUATING ICS Attack Surface}

\subsection{Performance of Autonomous Defense}


\section{LIMITATIONS AND CONCLUSION}











\bibliographystyle{IEEEtran}
\bibliography{ref}

\end{document}
