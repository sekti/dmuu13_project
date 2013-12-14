import java.util.Random;
import java.util.Map;
import java.util.HashMap;
import org.rlcommunity.rlglue.codec.AgentInterface;
import org.rlcommunity.rlglue.codec.types.RL_abstract_type;
import org.rlcommunity.rlglue.codec.types.Action;
import org.rlcommunity.rlglue.codec.types.Observation;
import org.rlcommunity.rlglue.codec.util.AgentLoader;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpec;
import org.rlcommunity.rlglue.codec.taskspec.ranges.IntRange;

public class SarsaHistoryAgent implements AgentInterface {

    private static final double defaultValue = 1;

    private Random rand = new Random();

    // Current and next state of agent
    private State state, nextState;
    
    private Action lastAction;
    
    private int discreteActionDims;
    private IntRange[] discreteActionRanges;
    private int numActions;
   
    private int[] indexSteps;

    // State-Action value function implemented as a map
    private Map<State, double[]> Q;    
    
    // Algoritm parameters
    private double alpha = 0.1;
    private double epsilon = 0.1;
    private double gamma;   

    public void agent_init(String taskSpecification) {
        TaskSpec ts = new TaskSpec(taskSpecification);

        /* Make sure that we can handle the problem as specified.
	 * It is assumed that the actions and observations consists of only discrete 
	 * vectors of integers.
	 */
	assert (ts.getNumContinuousActionDims() == 0);
	assert (ts.getNumContinuousObsDims() == 0);
	
	gamma = ts.getDiscountFactor();	

	// Get the ranges of all integers of a completely discrete action
	discreteActionDims = ts.getNumDiscreteActionDims();
	discreteActionRanges = new IntRange[discreteActionDims];
	indexSteps = new int[discreteActionDims];
	
	numActions = 1;
	for (int i = 0; i < discreteActionDims; i++) {
	    discreteActionRanges[i] = ts.getDiscreteActionRange(i);
	    numActions *= discreteActionRanges[i].getRangeSize();
	}

	indexSteps[0] = 1;
	for (int i = 1; i < discreteActionDims; i++)
	    indexSteps[i] = discreteActionRanges[i].getRangeSize() * indexSteps[i - 1];
	
	// Initialize the state-action value function Q
	Q = new HashMap<State, double[]>();

	// Test of the actionIndex and indexAction functions
	for (int i = 0; i < numActions; i++)
	    if (actionIndex(indexAction(i)) != i)
		System.out.printf("Test failed!\n");
     }
    
    public Action agent_start(Observation observation) {
	// Add the given observation if it does not already exists in the map
        state = new State(observation.duplicate());

	if (!Q.containsKey(state))	   
	    Q.put(state, defaultValues());	

	// Select the greedy action corresponding to the first state
        Action action = greedyAction(state);

	// Save the last action
        lastAction = action.duplicate();
        
        return action;
    }
  
    public Action agent_step(double reward, Observation observation) {
        //nextState = new State(observation.duplicate());
	nextState = state.next(lastAction, observation.duplicate());

	// Add the next state if it does not already exists in the map
	if (!Q.containsKey(nextState))	   
	    Q.put(nextState, defaultValues());	   
	
	// Select greedy action for the next state
	Action action = greedyAction(nextState);

	// SARSA learning
	double[] Q_s = Q.get(state);
	double Q_sa = Q_s[actionIndex(lastAction)];
        double Q_sprime_aprime = Q.get(nextState)[actionIndex(action)];
	Q_s[actionIndex(lastAction)] = Q_sa + alpha * (reward + gamma * Q_sprime_aprime - Q_sa);

        // Make a transition to the next state and save the action taken
	state = nextState;
        lastAction = action.duplicate();

        return action;
    }
   
    public void agent_end(double reward) {
	// SARSA learning (last step of episode)
	double[] Q_s = Q.get(state);
	double Q_sa = Q_s[actionIndex(lastAction)];
	Q_s[actionIndex(lastAction)] = Q_sa + alpha * (reward - Q_sa);
    }
   
    public void agent_cleanup() {      
    }

    public String agent_message(String message) {
        return "Agent does not handle any messages.";
    }

    /* Selects a random action with probability 1 - epsilon,
     * and the action with the highest value otherwise. 
     */
    private Action greedyAction(State state) {
	if (rand.nextDouble() <= epsilon)
	    return indexAction(rand.nextInt(numActions));
	else
	    return indexAction(maxAction(state));        
    }

    // Compute and return the best action index for a given state
    private int maxAction(State state) {	
	double[] Q_s = Q.get(state);
	
	int maxIndex = 0;
        for (int i = 1; i < numActions; i++)
            if (Q_s[i] > Q_s[maxIndex])
                maxIndex = i;
     
        return maxIndex;
    }

    // Compute the action index that may be used in a 1-dim array of action values
    private int actionIndex(Action a) {
	int index = 0;
	for (int i = 0; i < discreteActionDims; i++)
	    index += (a.getInt(i) - discreteActionRanges[i].getMin()) * indexSteps[i];

	return index;
    }

    // Construct an action corresponding to a specific index
    private Action indexAction(int index) {
	Action a = new Action(discreteActionDims, 0, 0);

	for (int i = discreteActionDims - 1; i >= 0; i--) {
	    int x = index / indexSteps[i];
	    index -= x * indexSteps[i];
	    a.setInt(i, x + discreteActionRanges[i].getMin());
	}

	return a;
    }

    private double[] defaultValues() {
	double[] values = new double[numActions];
	for (int i = 0; i < numActions; i++)
	    values[i] = defaultValue;

	return values;
    }

    public static void main(String[] args) {
        AgentLoader theLoader = new AgentLoader(new SarsaHistoryAgent());
        theLoader.run();
    }

    // A state consists of a finite history of observations and actions
    private class State {
	private static final int maxLength = 0;
	
	public int length;
	public Observation[] observations;
	public Action[] actions;

	public State(Observation o) {	    
	    this(0);
	    observations[0] = o;
	}
	
	public State(int length) {	    
	    this.length = length;
	    observations = new Observation[length + 1];
	    actions = new Action[length];	
	} 

	/* Construct and return the next state that is obtained 
	 * by appending an action and observation.
	 */    
	public State next(Action a, Observation o) {
	    State nextState;
	    
	    if (maxLength == 0) {
		nextState = new State(o);
	    }
	    
	    else if (length < maxLength) {
		nextState = new State(length + 1);
	    
		nextState.observations[0] = this.observations[0];
	    
		for (int i = 0; i < length; i++) {
		    nextState.actions[i] = this.actions[i];
		    nextState.observations[i + 1] = this.observations[i + 1];
		}

		nextState.actions[length] = a;
		nextState.observations[length + 1] = o;		
	    }
	    
	    else /* (length == maxLength) */ {
		nextState = new State(length);	    
		
		nextState.observations[0] = this.observations[1];

		for (int i = 0; i < length - 1; i++) {
		    nextState.actions[i] = this.actions[i + 1];
		    nextState.observations[i + 1] = this.observations[i + 2];
		}
		
		nextState.actions[length - 1] = a;
		nextState.observations[length] = o;
	    }

	    return nextState;
	    } 

	/* Equality comparison, required for using State objects as keys in maps.
	 * Two states are considered equal if and only if they are of the same length 
	 * and all their observations and actions are equal.
	 */
	@Override 
	public boolean equals(Object x) {
	    if (x == null)
		return false;

	    if (x == this) 
		return true;
	   
	    if (!(x instanceof State))
		return false;

	    State s = (State)x;

	    if(s.length != this.length)
		return false;	   

	    if (s.observations[0].compareTo(this.observations[0]) != 0) 
	    	return false;	  

	    for (int i = 0; i < this.length; i++)
		if (s.actions[i].compareTo(this.actions[i]) != 0 ||
		    s.observations[i + 1].compareTo(this.observations[i + 1]) != 0)
		    return false;
	    
	    return true;
	}

	@Override
	public int hashCode() {
	    int hash = 0;

	    hash += intSum(this.observations[0]);

	    for (int i = 0; i < length; i++) {
		hash += intSum(this.actions[i]);
		hash += intSum(this.observations[i + 1]);
	    }

	    return hash;
	}
	
	private int intSum(RL_abstract_type x) {
	    int sum = 0;
	    
	    for (int i = 0; i < x.getNumInts(); i++)
		sum += x.getInt(i);
	    
	    return sum;
	}
	    
    }

}
