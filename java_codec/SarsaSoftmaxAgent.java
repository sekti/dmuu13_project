import java.util.Random;
import org.rlcommunity.rlglue.codec.AgentInterface;
import org.rlcommunity.rlglue.codec.types.Action;
import org.rlcommunity.rlglue.codec.types.Observation;
import org.rlcommunity.rlglue.codec.util.AgentLoader;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpec;

public class SarsaSoftmaxAgent implements AgentInterface {

    private Random randGenerator = new Random();

    private Action lastAction;
    private Observation lastObservation;
    
    private double[][] valueFunction = null;
    
    private double alpha = 0.9;
    private double gamma;

    // The temperature in the Gibbs distribution used in action selection
    private double tau = 0.01;

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
        assert (ts.getDiscreteActionRange(0).getMin() == 0);
        
        numStates = ts.getDiscreteObservationRange(0).getMax() + 1;
        numActions = ts.getDiscreteActionRange(0).getMax() + 1;
        
        gamma = ts.getDiscountFactor();	

        valueFunction = new double[numActions][numStates];
    }
    
    public Action agent_start(Observation observation) {
        int newActionInt = softmaxAction(observation.getInt(0));

        /**
         * Create a structure to hold 1 integer action
         * and set the value
         */
        Action returnAction = new Action(1, 0, 0);
        returnAction.intArray[0] = newActionInt;       
        
        lastAction = returnAction.duplicate();
        lastObservation = observation.duplicate();

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
        int newStateInt = observation.getInt(0);
        int lastStateInt = lastObservation.getInt(0);
        int lastActionInt = lastAction.getInt(0);

        int newActionInt = softmaxAction(newStateInt);      

        double Q_sa = valueFunction[lastActionInt][lastStateInt];
        double Q_sprime_aprime = valueFunction[newActionInt][newStateInt];
        double new_Q_sa = Q_sa + alpha * (reward + gamma * Q_sprime_aprime - Q_sa);
        
        valueFunction[lastActionInt][lastStateInt] = new_Q_sa;
        
        /* Creating the action a different way to showcase variety */
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
        double new_Q_sa = Q_sa + alpha * (reward - Q_sa);

        valueFunction[lastActionInt][lastStateInt] = new_Q_sa;

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

    /* Action selection using the Gibbs distribution.
     * Action k is selected out of a total of n actions at state s with probability
     * e^(Q(s, k) / tau) / (e^(Q(s, 1) / tau) + ... + e^(Q(s, n) / tau)),
     * where Q(s, k) is the estimated value of taking action k at state s and tau is the "temperature".
     */
    private int softmaxAction(int state) {
        double[] exps = new double[numActions];
        
        double sum = 0;
        for (int a = 0; a < numActions; a++) {
            exps[a] = Math.exp(valueFunction[a][state] / tau);
            sum += exps[a];
        }
       
        double r = randGenerator.nextDouble();
        double limit = 0;
        for (int a = 0; a < numActions; a++) {
            limit += exps[a] / sum;
            if (r < limit)
                return a;
        }

        return numActions - 1;
    }

    /**
     * This is a trick we can use to make the agent easily loadable.  Using this
     * trick you can directly execute the class and it will load itself through
     * AgentLoader and connect to the rl_glue server.
     * @param args
     */
    public static void main(String[] args) {
        AgentLoader theLoader = new AgentLoader(new SarsaSoftmaxAgent());
        theLoader.run();
    }
}
