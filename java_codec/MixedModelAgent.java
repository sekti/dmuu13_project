import java.util.Random;
import java.util.Map;
import java.util.HashMap;
import org.rlcommunity.rlglue.codec.AgentInterface;
import org.rlcommunity.rlglue.codec.types.Action;
import org.rlcommunity.rlglue.codec.types.Observation;
import org.rlcommunity.rlglue.codec.util.AgentLoader;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpec;

public class MixedModelAgent implements AgentInterface {
   
    /* Generics cannot be used in arrays, so strip it away
     * to have a non-generic class */
    private class StateMap extends HashMap<Integer,Integer> {
        public StateMap() {
            super();
        }
    }
    
    private Random rand = new Random();
   
    boolean freezeLearning;
    private double epsilon(int state) {
        return freezeLearning ? 0 : Math.pow(visitsSum[state], -0.8);
    }
    
    private double delta = 0.01;
    private double gamma;
    private static final int BATCH_SIZE = 1; /* increase this if value iteration is too slow */
    
    // The number of states
    private int S;

    // The number of actions
    private int A;   

    // The current estimate of the transition probabilities of the unknown MDP
    private StateMap[][] p;

    // The current estimates of the expected reward obtained for a certain state-action pair.
    // rD[s][a] is the estimated reward of performing action a in state s, given that the MDP is deterministic.
    // rS[s][a] is the estimated reward of performing action a in state s, given that the MDP is stochastic.
    private double[][] rD, rS;

    // Counters for the number of times each state-action pair has been visited.
    // visits[s][a] is the number of times action a has been taken in state s.
    // visits[s] is the total number of times some action has been taken in state s.
    private int[][] visits;
    private int[] visitsSum;

    // Boolean which is true if and only if the MDP cannot be deterministic
    boolean mdpIsStochastic;

    // Mapping of the deterministic state transitions observed so far.
    // ds[s][a] is the deterministic next state given that action a was taken in state s.
    private int[][] ds;

    // The sFactor determines the current subjective probability that the MDP is stochastic vs. deterministic.
    // As the history of observations grows in length, the sFactor will generally tend toward 0 as long as the 
    // history is consistent with a deterministic model.
    // At each time step, the probabilities pD and pS that we have a deterministic or a stochastic MDP are given as
    // (I) If the history is consistent with both a deterministic and stochastic model, then
    //     pD = 1 / (1 + sFactor)
    //     pS = 1 / (1 + (1 / sFactor))
    // (II) If the history is not consistent with a deterministic model, then
    //      pD = 0 and pS = 1.
    // The prior probabilities are pD = pS = 1 / 2.
    double sFactor;
    double pD, pS;    
    
    // The optimal stationary Markov policy based on the current information.
    // pi[s] is the action to take in state s.
    private int[] pi;

    // The current value function corresponding to the optimal policy.
    // v[s] is the estimated optimal total expected reward of state s.
    private double[] v;

    // The current state and action
    private int state, action;
    private int steps;

    public void agent_init(String taskSpecification) {
        TaskSpec ts = new TaskSpec(taskSpecification);

        // Make sure there are no continuous variables in our task
        assert (ts.getNumContinuousActionDims() == 0);
        assert (ts.getNumContinuousObsDims() == 0);

        // Get the discount factor for the task
        gamma = ts.getDiscountFactor();
        gamma = Math.min(gamma, 0.95);

        // The the total number of states and actions
        S = ts.getDiscreteObservationRange(0).getMax() + 1;
        A = ts.getDiscreteActionRange(0).getMax() + 1;

        // Initialize the transition probabilities
        p = new StateMap[S][A];

        // Initialize the expected reward estimates
        rD = new double[S][A];
	rS = new double[S][A];
        for (int s = 0; s < S; s++) {
            for (int a = 0; a < A; a++) {
                rD[s][a] = ts.getRewardMax();
		rS[s][a] = ts.getRewardMax();
                p[s][a] = new StateMap();
            }
        }
        // Initialize the number of visits to each state-action pair
        visits = new int[S][A];
        visitsSum = new int[S];

        mdpIsStochastic = false;
	
	ds = new int[S][A];

	sFactor = 1;

	// "Uniform" prior. We that a deterministic MDP is just as likely as a stochastic one.
	pD = 0.5;
	pS = 0.5;

        // Initialize the policy and value function
        pi   = new int[S];
        v    = new double[S];
        newv = new double[S];
    }
    
    public Action agent_start(Observation observation) {
        state = observation.getInt(0);	
        action = chooseAction(state);
        visits[state][action]++;
        visitsSum[state]++;
        
        // Return the choosen action
        Action returnAction = new Action(1, 0, 0);   
        returnAction.setInt(0, action);
        steps++;
        return returnAction;
    }
   
    public Action agent_step(double reward, Observation observation) {
        int nextState = observation.getInt(0);
			
	//// Handle the deterministic part of the model
	if (!mdpIsStochastic) {
	    if (visits[state][action] == 1) {
		ds[state][action] = nextState;
		rD[state][action] = reward;
	    }
	    else if (visits[state][action] > 1) {
		sFactor *= visits[state][action] / (S + visits[state][action] - 1);
		pD = 1 / (1 + sFactor);
		pS = 1 - pD;
	       
		if (nextState != ds[state][action] || reward != rD[state][action]) {
		    mdpIsStochastic = true;
		    pD = 0;
		    pS = 1;
		}
	    }	   
	}	

	//// Handle the stochastic part of the model

	// Update our current beliefs about the transition probabilities
        Integer times = p[state][action].get(nextState);
        p[state][action].put(nextState, times == null ? 1 : times + 1);
        
        // Update our current beliefs about the expected rewards
        rS[state][action] = ((visits[state][action] - 1) * rS[state][action] + reward) / visits[state][action];

        // Find an optimal policy for the estimated MDP via value iteration
        if (steps % BATCH_SIZE == 0) {
            //don't do it every time, it is too expensive	   
            valueIteration();
        }

        // Update state
        state = nextState;

        // Choose an action w.r.t. the current policy
        action = chooseAction(state);
        visits[state][action]++;       	
        visitsSum[state]++;
        
        // Return the choosen action
        Action returnAction = new Action(1, 0, 0);
        returnAction.setInt(0, action);
        steps++;
        
        return returnAction;
    }
    
    public void agent_end(double reward) {
	//// Handle the deterministic part of the model
	if (!mdpIsStochastic) {
	    if (visits[state][action] == 1) {
		rD[state][action] = reward;
	    }
	    else if (visits[state][action] > 1) {
		sFactor *= visits[state][action] / (S + visits[state][action] - 1);
		pD = 1 / (1 + sFactor);
		pS = 1 - pD;
	       
		if (reward != rD[state][action]) {
		    mdpIsStochastic = true;
		    pD = 0;
		    pS = 1;
		}
	    }	   
	}	

	//// Handle the stochastic part of the model
	
        // Update our current beliefs about the expected rewards	
        rS[state][action] = ((visits[state][action] - 1) * rS[state][action] + reward) / visits[state][action];
    }

    public void agent_cleanup() {
    }

    public String agent_message(String message) {
        if (message.equals("freeze learning")) {
            freezeLearning = true;
        } else if (message.equals("unfreeze learning")) {
            freezeLearning = false;
	} else if (message.equals("what is your name?")) {
	    return "SuperModelAgent";
        } else {
            System.out.println("Unhandled Message received: " + message);
        }
        
        return "Agent does not handle any messages.";
    }    
   
    private double arraySum(double[] v) {
	double sum = 0;
	for (double d : v)
            sum += d;
       
	return sum;
    }

    private double expectedReward(int s, int a, double[] v) {

	// Precompute the sum of v over all states, since we need it twice
	double vSum = arraySum(v);

        /* Initialize to sum of v over all states due to prior */
	double stochasticFutureReward = vSum;
        
	/* Sum up the values of all the states in which we ended up
	 * from s doing a in the past (with multiplicities). */
	for (Map.Entry<Integer, Integer> entry : p[s][a].entrySet()) {
	    /*                times it happend      stateIndex */
	    stochasticFutureReward += entry.getValue() * v[entry.getKey()];
	}        
        
	stochasticFutureReward /= S + visits[s][a];
	
	double deterministicFutureReward = visits[s][a] == 0 ? (vSum / S) : v[ds[s][a]];
	
	// Return future expected reward weighted according to our current model beliefs
        return 
	    pD * (rD[s][a] + gamma * deterministicFutureReward) +
	    pS * (rS[s][a] + gamma * stochasticFutureReward);
    }
    
    private double maxOfArray(double[] v) {
        double max = v[0];
        
        for (double d : v) {
            max = Math.max(max, d);
        }
        
        return max;
    }
 
    private double supNorm(double[] v1, double[] v2) {
        double res = 0.0;
        
        for (int i = 0; i < v1.length; i++)
            res = Math.max(Math.abs(v1[i] - v2[i]), res);

        return res;
    }
    
    private double[] newv; //allocate once.
    private void valueIteration() {       
        double[] oldv   = null;                 //temporary
        
        do {	    
            for (int s = 0; s < S; s++) {
                /* for some reason using Double.NEGATIVE_INFINITY here
                 * will break the algorithm. I do not understand that.
                 * Can Math.max not handle infinity? */
                newv[s] = expectedReward(s, 0, v);
                pi  [s] = 0;
                
		// The below is only necessary if we have visited the state s at least once.
		// Otherwise, the maximization over actions just gives the initialization values above.
		if (visitsSum[s] > 0)
		    for (int a = 1; a < A; a++) {
			double r = expectedReward(s, a, v);
                    
			if (r > newv[s]) {
			    newv[s] = r;
			    pi  [s] = a;
			}
		    }
            }	    
            
            oldv = v;
            v = newv;
        } while (supNorm(oldv, v) > delta * (1 - gamma) / (2 * gamma));
    }
    
    public int chooseAction(int state) {
        if (rand.nextDouble() <= epsilon(state)) {
            /* pick random action: */
            /* prefer those, that we took rarely so far */
            double sum = 0.0;
            for (int a = 0; a < A; ++a) {
                sum += 1.0 / (visits[state][a] + 1);
            }
            
            double r = rand.nextDouble() * sum;
            
            double limit = 0;
            for(int a = 0; a < A; ++a) {
                limit += 1.0 / (visits[state][a] + 1);
                
                if (limit >= r) {
                    return a;
                }
            }
            assert(false);
            return A - 1;
        } else
            return pi[state];
    }

    public static void main(String[] args) {
        AgentLoader theLoader = new AgentLoader(new MixedModelAgent());
        theLoader.run();
    }
}
