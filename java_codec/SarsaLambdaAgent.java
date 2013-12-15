import java.util.Random;
import org.rlcommunity.rlglue.codec.AgentInterface;
import org.rlcommunity.rlglue.codec.types.Action;
import org.rlcommunity.rlglue.codec.types.Observation;
import org.rlcommunity.rlglue.codec.util.AgentLoader;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpec;

public class SarsaLambdaAgent implements AgentInterface {

    private Random randGenerator = new Random();

    private Action lastAction;
    private Observation lastObservation;
    
    private double[][] valueFunction = null;
    private double[][] trace = null;

    private double alpha = 0.1;
    private double epsilon = 0.1;
    private double gamma = 1.0;
    private double lambda = 0.6;

    private int numActions = 0;
    private int numStates = 0;    

    public void agent_init(String taskSpecification) {
        TaskSpec ts = new TaskSpec(taskSpecification);

        /* Make sure that we can handle the problem as specified.
         * It is assumed that the actions and observations consists of only discrete 
         * vectors of integers.
         */
        assert (ts.getNumContinuousActionDims() == 0);
        assert (ts.getNumContinuousObsDims() == 0);
        
        numStates  = ts.getDiscreteObservationRange(0).getMax() + 1;
        numActions = ts.getDiscreteActionRange     (0).getMax() + 1;

        gamma = ts.getDiscountFactor();	

        valueFunction = new double[numActions][numStates];

        // This will initialize the trace to 0 for all states and actions
        trace = new double[numActions][numStates];
    }
    
    public Action agent_start(Observation observation) {
        int newActionInt = egreedy(observation.getInt(0));

        /**
         * Create a structure to hold 1 integer action
         * and set the value
         */
        Action returnAction = new Action(1, 0, 0);
        returnAction.intArray[0] = newActionInt;       
    
        lastAction = returnAction.duplicate();
        lastObservation = observation.duplicate();
        
        /* Reset the trace because we start anew! */
        trace = new double[numActions][numStates];
        
        return returnAction;
    }

    /**
     * Choose an action e-greedily from the value function and store the action
     * and observation.  Update the valueFunction entry for the last
     * state,action pair.
     * @param reward
     * @param observation
     * @return
     */
    public Action agent_step(double reward, Observation observation) {
        int newStateInt   = observation.getInt(0);
        int lastStateInt  = lastObservation.getInt(0);
        int lastActionInt = lastAction.getInt(0);
        int newActionInt  = egreedy(newStateInt);      

        double Q_sa = valueFunction[lastActionInt][lastStateInt];
        double Q_sprime_aprime = valueFunction[newActionInt][newStateInt];
    
        double delta = reward + gamma * Q_sprime_aprime - Q_sa;
        
        trace[lastActionInt][lastStateInt] += 1;
        
        for (int i = 0; i < numStates; i++) {
            for (int j = 0; j < numActions; j++) {
                valueFunction[j][i] += alpha * delta * trace[j][i];
                trace[j][i] *= gamma * lambda;
            }
        }

        Action returnAction = new Action();
        returnAction.intArray = new int[]{newActionInt};

        lastAction = returnAction.duplicate();
        lastObservation = observation.duplicate();

        return returnAction;
    }

    /**
     * The episode is over, learn from the last reward that was received.
     * @param reward
     */
    public void agent_end(double reward) {
        int lastStateInt = lastObservation.getInt(0);
        int lastActionInt = lastAction.getInt(0);

        double Q_sa = valueFunction[lastActionInt][lastStateInt];

        double delta = reward - Q_sa;
        
        trace[lastActionInt][lastStateInt] += 1;	

        for (int i = 0; i < numStates; i++) {
            for (int j = 0; j < numActions; j++) {
                valueFunction[j][i] += alpha * delta * trace[j][i];
                trace[j][i] *= gamma * lambda;
            }
        }	

        lastObservation = null;
        lastAction = null;
    }

    /**
     * Release memory that is no longer required/used.
     */
    public void agent_cleanup() {
        lastAction = null;
        lastObservation = null;
        valueFunction = null;
    }

    public String agent_message(String message) {
        return "Agent does not handle any messages.";
    }

    /**
     *
     * Selects a random action with probability 1 - epsilon,
     * and the action with the highest value otherwise.  This is a
     * quick'n'dirty implementation, it does not do tie-breaking.

     * @param theState
     * @return
     */
    private int egreedy(int state) {
        if (randGenerator.nextDouble() <= epsilon)
            return randGenerator.nextInt(numActions);
        else
            return maxAction(state);        
    }

    /* Compute and return the best action for a given state */
    private int maxAction(int state) {	
        int maxIndex = 0;

        for (int a = 1; a < numActions; a++) {
            if (valueFunction[a][state] > valueFunction[maxIndex][state]) {
                maxIndex = a;
            }
        }

        return maxIndex;
    } 

    /**
     * This is a trick we can use to make the agent easily loadable.  Using this
     * trick you can directly execute the class and it will load itself through
     * AgentLoader and connect to the rl_glue server.
     * @param args
     */
    public static void main(String[] args) {
        AgentLoader theLoader = new AgentLoader(new SarsaLambdaAgent());
        theLoader.run();
    }
}
