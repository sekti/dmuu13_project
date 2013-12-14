import java.util.Random;
import org.rlcommunity.rlglue.codec.AgentInterface;
import org.rlcommunity.rlglue.codec.types.Action;
import org.rlcommunity.rlglue.codec.types.Observation;
import org.rlcommunity.rlglue.codec.util.AgentLoader;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpec;

public class ModelAgent implements AgentInterface {

    private Random rand = new Random();
   
    private double epsilon = 0.05;
    private double delta = 0.01;
    private double gamma;
    
    // The number of states
    private int S;

    // The number of actions
    private int A;   

    // The current estimate of the transition probabilities of the unknown MDP.
    // p[s][a][j] is the estimated probability of making a transition from state s to state j when action a is taken.
    private double[][][] p;

    // The current estimate of the expected reward obtained for a certain state-action pair.
    // r[s][a] is the estimated reward of performing action a in state s.
    private double[][] r;

    // Counters for the number of times each state-action pair has been visited.
    // visits[s][a] is the number of times action a has been taken in state s.
    private int[][] visits;

    // The optimal stationary Markov policy based on the current information.
    // pi[s] is the action to take in state s.
    private int[] pi;

    // The current value function corresponding to the optimal policy.
    // v[s] is the estimated optimal total expected reward of state s.
    private double[] v;

    // The current state and action
    private int state, action;

    public void agent_init(String taskSpecification) {
        TaskSpec ts = new TaskSpec(taskSpecification);

	// Make sure there are no continuous variables in our task
	assert (ts.getNumContinuousActionDims() == 0);
	assert (ts.getNumContinuousObsDims() == 0);

	// Get the discount factor for the task
	gamma = ts.getDiscountFactor();	

	// The the total number of states and actions
	S = ts.getDiscreteObservationRange(0).getMax() + 1;
        A = ts.getDiscreteActionRange(0).getMax() + 1;

	// Initialize the estimates for the transition probabilities
	p = new double[S][A][S];
	for (int s = 0; s < S; s++)
	    for (int a = 0; a < A; a++)
		for (int j = 0; j < S; j++)
		    p[s][a][j] = 1 / S;

	// Initialize the expected reward estimates
	r = new double[S][A];
	for (int s = 0; s < S; s++)
	    for (int a = 0; a < A; a++)
		r[s][a] = 10; // TODO: what value is best to use here?

	// Initialize the number of visits to each state-action pair
	visits = new int[S][A];

	// Initialize the policy and value function
	pi = new int[S];
	v = new double[S];	
    }
    
    public Action agent_start(Observation observation) {
	state = observation.getInt(0);	
        action = chooseAction(state);
	visits[state][action]++;

	// Return the choosen action
        Action returnAction = new Action(1, 0, 0);   
	returnAction.setInt(0, action);
        return returnAction;
    }
   
    public Action agent_step(double reward, Observation observation) {
	int nextState = observation.getInt(0);

	// Update our current beliefs about the transition probabilities
	double alphaSum = S + visits[state][action];
	for (int j = 0; j < S; j++)	    	   
	    p[state][action][j] = (alphaSum - 1) * p[state][action][j] / alphaSum;

	p[state][action][nextState] += 1 / alphaSum;
       
	// Update our current beliefs about the expected rewards
	r[state][action] = ((visits[state][action] - 1) * r[state][action] + reward) / visits[state][action];

	// Find an optimal policy for the estimated MDP via value iteration
	valueIteration();

	// Update state
	state = nextState;

	// Choose an action w.r.t. the current policy
	action = chooseAction(state);
	visits[state][action]++;       	

	// Return the choosen action
	Action returnAction = new Action(1, 0, 0);
        returnAction.setInt(0, action);
        return returnAction;
    }
    
    public void agent_end(double reward) {
	// Update our current beliefs about the expected rewards	
	r[state][action] = ((visits[state][action] - 1) * r[state][action] + reward) / visits[state][action];
    }

    public void agent_cleanup() {
    }

    public String agent_message(String message) {
        return "Agent does not handle any messages.";
    }    

    private double expectedReward(int s, int a, double[] v) {
	double futureReward = 0;
	for (int j = 0; j < S; j++)
	    futureReward += p[s][a][j] * v[j];		   
	
        return r[s][a] + gamma * futureReward;
    }
    
    private double maxOfArray(double[] v1) {
	double max = v1[0];
	
	for (int i = 1; i < v1.length; i++)
	    max = Math.max(max, v1[i]);

	return max;
    }
 
    private double supNorm(double[] v1, double[] v2) {
	double[] d = new double[v1.length];

	for (int i = 0; i < v1.length; i++)
	    d[i] = Math.abs(v1[i] - v2[i]);

	return maxOfArray(d);
    }

    private void valueIteration() {       
	double[] reward = new double[A];

       	double[] w = v;
	do {
	    v = w;
	    w = new double[v.length];

	    for (int s = 0; s < S; s++) {
		
		for (int a = 0; a < A; a++)
		    reward[a] = expectedReward(s, a, v);
		
		w[s] = maxOfArray(reward);
	    }
	    
	} while (supNorm(w, v) > delta * (1 - gamma) / (2 * gamma));
	
	v = w;
	
	for (int s = 0; s < S; s++) {
	    int bestAction = 0;	  
	    for (int a = 0; a < A; a++) {	
		reward[a] = expectedReward(s, a, v);
		if (reward[a] > reward[bestAction])
		    bestAction = a;
	    }

	    pi[s] = bestAction;
	}
    }  
    
    public int chooseAction(int state) {
	if (rand.nextDouble() <= epsilon)
	    return rand.nextInt(A);
	else
	    return pi[state];
    }

    public static void main(String[] args) {
        AgentLoader theLoader = new AgentLoader(new ModelAgent());
        theLoader.run();
    }
}
