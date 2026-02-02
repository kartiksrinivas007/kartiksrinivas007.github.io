---
title: "Family"
date: 2026-02-01
draft: false
description: "Family link tree."
type: "page"
---

{{< mermaid >}}
%%{init: { "flowchart": { "useMaxWidth": false, "padding": 12, "nodeSpacing": 100, "rankSpacing": 120 }, "themeVariables": { "fontSize": "16px", "lineColor": "#64748b" } }}%%
flowchart TB
  %% Maternal grandparents
  MH[Maheendran]:::person
  RM[Ramani]:::person

  %% Maheendran's parents and siblings
  VPP["Velukutty & Gauri"]:::group
  C1[Ambika]:::person
  C2[Sumitra]:::person
  C3["Suchitra (Appi)"]:::person
  C4[Sivaji]:::person
  C5["Saratchandran (Anni)"]:::person
  C6[Satishan]:::person
  C7[Leela]:::person
  C8[Manorama]:::person
  C9["Hema (Amani)"]:::person
  C10["Jayadeep (London)"]:::person

  %% Ramani's parents and siblings
  RPP["Kartiyani & Vava"]:::group
  RS1[Sugadhan]:::person
  RS2[Ammavan]:::person
  RS3[Babu]:::person
  RS4[Sneha]:::person
  RS5[Vijayama]:::person
  RS6[Chandrika]:::person
  RS7[Thangamani]:::person
  RS8[Sundarambal]:::person
  RS9[Veliammavan]:::person
  RS10[Mutheyamma]:::person

  %% Paternal grandparents (two marriages)
  SD[Sudhakaran]:::person
  GT[Geeta]:::person
  VJ[Vijaya]:::person

  %% Parents of Sudhakaran, Raju, Appu
  RYP["Raghavan & Yashodham"]:::group

  %% Sudhakaran's brothers, their children and grandchildren
  RJ[Raju]:::person
  AP[Appu]:::person
  VJN[Vijayan]:::person
  AS[Asokan]:::person
  LX[Laxmi]:::person
  HM[Hema]:::person
  ACH[Achu]:::person
  AMV[Ammu]:::person
  SH[Sheril]:::person
  DP[Deepa]:::person
  DL[Dilip]:::person
  GN[Ganga]:::person
  PD[Pradeep]:::person
  AK[Akshath]:::person
  DV[Divya]:::person
  SN["Sundari (Remya)"]:::person
  KT["Kuttu (Rakesh)"]:::person
  VN[Vineeth]:::person
  SRN[Sreenika]:::person
  PR[Priya]:::person
  VR[Viraj]:::person
  RS[Rishi]:::person

  %% Parents
  SM[Smitha]:::person
  SR[Srinivas]:::person

  %% Maternal side siblings
  SMB[Jayanth]:::person
  NT[Neethu]:::person
  SJ[Sreeja]:::person
  AD[Advait]:::person

  %% Paternal side siblings
  SRB1[Sudhir]:::person
  SRB2["&emsp;Kumar&emsp;"]:::person
  VS[Vasavi]:::person
  VV[Vaishnavi]:::person
  KV[Kavitha]:::person
  AM["Ammu (Reema)"]:::person
  OM[Om]:::person

  %% Kids
  K[Kartik]:::person
  HR[Hrithik]:::person

  %% Edges: grandparents -> parents
  MH --> SM
  RM --> SM

  SD --> SR
  VJ --> SR

  %% Edges: Sudhakaran's parents -> his brothers
  RYP --> SD
  RYP --> RJ
  RYP --> AP
  RYP --> VJN
  RYP --> AS

  %% Edges: Maheendran's parents -> children
  VPP --> MH
  VPP --> C1
  VPP --> C2
  VPP --> C3
  VPP --> C4
  VPP --> C5
  VPP --> C6
  VPP --> C7
  VPP --> C8
  VPP --> C9
  VPP --> C10

  %% Edges: Ramani's parents -> children
  RPP --> RM
  RPP --> RS1
  RPP --> RS2
  RPP --> RS3
  RPP --> RS4
  RPP --> RS5
  RPP --> RS6
  RPP --> RS7
  RPP --> RS8
  RPP --> RS9
  RPP --> RS10

  %% Parents -> kids
  SM --> K
  SR --> K
  SM --> HR
  SR --> HR

  %% Parent siblings
  MH --> SMB
  RM --> SMB

  SMB --- NT
  SMB --- SJ
  SMB --> AD
  NT --> AD

  %% Paternal grandparents marriages / in-law connections
  RYP --- VJ
  SD --- GT
  SD --- VJ

  SD --> SRB1
  GT --> SRB1
  SD --> SRB2
  GT --> SRB2

  %% Sudhir and Kumar families
  SRB2 --- VS
  SRB1 --- KV

  SRB2 --> VV
  VS --> VV

  SRB1 --> AM
  KV --> AM
  SRB1 --> OM
  KV --> OM

  %% Raju and Appu descendants
  AP --- LX
  AP --> DP
  AP --> DL
  LX --> DP
  LX --> DL

  RJ --- SH
  RJ --> SN
  SH --> SN
  RJ --> KT
  SH --> KT

  SN --- VN
  SN --> SRN
  VN --> SRN

  KT --- PR
  KT --> VR
  PR --> VR
  KT --> RS
  PR --> RS

  %% Dilip and Deepa families
  DL --- GN
  DP --- PD
  DP --> AK
  PD --> AK
  DP --> DV
  PD --> DV

  %% Vijayan family
  VJN --- HM
  VJN --> ACH
  HM --> ACH
  VJN --> AMV
  HM --> AMV

  %% Clickable links to family pages
  click MH "/family/maheendran/" "Maheendran"
  click RM "/family/remya/" "Remya"
  click C1 "/family/ambika/" "Ambika"
  click C2 "/family/sumitra/" "Sumitra"
  click C3 "/family/suchitra-appi/" "Suchitra (Appi)"
  click C4 "/family/sivaji/" "Sivaji"
  click C5 "/family/saratchandran-anni/" "Saratchandran (Anni)"
  click C6 "/family/satishan/" "Satishan"
  click C7 "/family/leela/" "Leela"
  click C8 "/family/manorama/" "Manorama"
  click C9 "/family/hema-amani/" "Hema (Amani)"
  click C10 "/family/jayadeep-london/" "Jayadeep (London)"
  click RS1 "/family/sugadhan/" "Sugadhan"
  click RS2 "/family/ammavan/" "Ammavan"
  click RS3 "/family/babu/" "Babu"
  click RS4 "/family/sneha/" "Sneha"
  click RS5 "/family/vijayama/" "Vijayama"
  click RS6 "/family/chandrika/" "Chandrika"
  click RS7 "/family/thangamani/" "Thangamani"
  click RS8 "/family/sundarambal/" "Sundarambal"
  click RS9 "/family/veliammavan/" "Veliammavan"
  click RS10 "/family/mutheyamma/" "Mutheyamma"
  click SD "/family/paternal-grandparents/" "Sudhakaran"
  click GT "/family/paternal-grandparents/" "Geeta"
  click VJ "/family/paternal-grandparents/" "Vijaya"
  click RJ "/family/raju/" "Raju"
  click AP "/family/appu/" "Appu"
  click VJN "/family/vijayan/" "Vijayan"
  click AS "/family/asokan/" "Asokan"
  click LX "/family/laxmi/" "Laxmi"
  click HM "/family/hema/" "Hema"
  click ACH "/family/achu/" "Achu"
  click AMV "/family/ammu/" "Ammu"
  click SH "/family/sheril/" "Sheril"
  click GN "/family/ganga/" "Ganga"
  click PD "/family/pradeep/" "Pradeep"
  click AK "/family/akshath/" "Akshath"
  click DV "/family/divya/" "Divya"
  click SM "/family/smitha/" "Smitha"
  click SR "/family/srinivas/" "Srinivas"
  click SMB "/family/smitha-brother/" "Smitha's Brother"
  click SRB1 "/family/srinivas-brother-1/" "Srinivas' Brother 1"
  click SRB2 "/family/srinivas-brother-2/" "Srinivas' Brother 2"
  click VS "/family/vasavi/" "Vasavi"
  click VV "/family/vaishnavi/" "Vaishnavi"
  click KV "/family/kavitha/" "Kavitha"
  click AM "/family/ammu-reema/" "Ammu (Reema)"
  click OM "/family/om/" "Om"
  click K "/family/kartik/" "Kartik"
  click HR "/family/hrithik/" "Hrithik"

  classDef group fill:#f4f4f5,stroke:#f4f4f5,stroke-width:1px,rx:8px,ry:8px;
  classDef person fill:#0f172a,stroke:#0f172a,color:#e5e7eb,stroke-width:1px,rx:8px,ry:8px,padding:0px 28px;
{{< /mermaid >}}
