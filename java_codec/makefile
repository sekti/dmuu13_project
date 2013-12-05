CP    =-cp ".;./JavaRLGlueCodec.jar;"
EXP   =SampleExperiment
AGENT =SampleSarsaAgent
ENV   =SampleMinesEnvironment

all : $(EXP).class $(AGENT).class $(ENV).class

run :
	java $(CP) $(EXP) & java $(CP) $(ENV) & java $(CP) $(AGENT) & ./../core/rl_glue.exe

%.class : %.java
	javac $(CP) $*.java