import java.util.Random;
import org.rlcommunity.rlglue.codec.AgentInterface;
import org.rlcommunity.rlglue.codec.types.Action;
import org.rlcommunity.rlglue.codec.types.Observation;
import org.rlcommunity.rlglue.codec.util.AgentLoader;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpec;

/* This is our Agent */
public class Agent implements AgentInterface {
    private Random randGenerator = new Random();
    private Action lastAction;
    private Observation lastObservation;
    private double[][] Q = null;
    private double alpha = 0.1;
    private double epsilon = 0.1;
    private double gamma = 1.0;
    private int numActions = 0;
    private int numStates = 0;

    /**
     * Parse the task spec, make sure it is only 1 integer observation and
     * action, and then allocate the Q.
     *
     * @param taskSpecification
     */
    public void agent_init(String taskSpecification) {
        TaskSpec theTaskSpec = new TaskSpec(taskSpecification);

        /* Lots of assertions to make sure that we can handle this problem.  */
        assert (theTaskSpec.getNumDiscreteObsDims() == 1);
        assert (theTaskSpec.getNumContinuousObsDims() == 0);
        assert (!theTaskSpec.getDiscreteObservationRange(0).hasSpecialMinStatus());
        assert (!theTaskSpec.getDiscreteObservationRange(0).hasSpecialMaxStatus());
        numStates = theTaskSpec.getDiscreteObservationRange(0).getMax() + 1;

        assert (theTaskSpec.getNumDiscreteActionDims() == 1);
        assert (theTaskSpec.getNumContinuousActionDims() == 0);
        assert (!theTaskSpec.getDiscreteActionRange(0).hasSpecialMinStatus());
        assert (!theTaskSpec.getDiscreteActionRange(0).hasSpecialMaxStatus());
        numActions = theTaskSpec.getDiscreteActionRange(0).getMax() + 1;

        gamma=theTaskSpec.getDiscountFactor();

        Q = new double[numActions][numStates];
    }

    /**
     * Choose an action e-greedily from the value function and store the action
     * and observation.
     * @param observation
     * @return
     */
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

        return returnAction;
    }

    /**
     * Choose an action e-greedily from the value function and store the action
     * and observation.  Update the Q entry for the last
     * state,action pair.
     * @param reward
     * @param observation
     * @return
     */
    public Action agent_step(double reward, Observation observation) {
        int newStateInt = observation.getInt(0);
        int lastStateInt = lastObservation.getInt(0);
        int lastActionInt = lastAction.getInt(0);

        int newActionInt = egreedy(newStateInt);

        double Q_sa = Q[lastActionInt][lastStateInt];
        double Q_sprime_aprime = Q[newActionInt][newStateInt];

        double new_Q_sa = Q_sa + alpha * (reward + gamma * Q_sprime_aprime - Q_sa);
        /*	Only update the value function if the policy is not frozen */
        Q[lastActionInt][lastStateInt] = new_Q_sa;

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

        double Q_sa = Q[lastActionInt][lastStateInt];
        double new_Q_sa = Q_sa + alpha * (reward - Q_sa);

        /*	Only update the value function if the policy is not frozen */
        Q[lastActionInt][lastStateInt] = new_Q_sa;
        
        lastObservation = null;
        lastAction = null;
    }

    /**
     * Release memory that is no longer required/used.
     */
    public void agent_cleanup() {
        lastAction = null;
        lastObservation = null;
        Q = null;
    }

    /**
     * This agent responds to some simple messages for freezing learning and
     * saving/loading the value function to a file.
     * @param message
     * @return
     */
    public String agent_message(String message) {
        return "We don't care for your messages.";
    }

    /**
     *
     * Selects a random action with probability 1-epsilon,
     * and the action with the highest value otherwise.  This is a
     * quick'n'dirty implementation, it does not do tie-breaking.

     * @param theState
     * @return
     */
    private int egreedy(int theState) {
        if (randGenerator.nextDouble() <= epsilon) {
            return randGenerator.nextInt(numActions);
        }

        /*otherwise choose the greedy action*/
        int maxIndex = 0;
        for (int a = 1; a < numActions; a++) {
            if (Q[a][theState] > Q[maxIndex][theState]) {
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
        AgentLoader theLoader = new AgentLoader(new SampleSarsaAgent());
        theLoader.run();
    }
}
