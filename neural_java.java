//Hi this is basic neural network with one hidden layer
//Actually implemented on behalf of of devloping a neumeric analysis library in java
/*This neural net will learn the toy code 
  ---------
  |0 0 1|0|
  |     | |
  |0 1 1|1|
  |     | |
  |1 0 1|1|
  |     | |
  |1 1 1|0|
*/
import numja.goofi;
public class neural_java{
             public static double[][] nonlin(double[][] x){
             //This is  our nonlinearity function
                              return goofi.divmat(1,goofi.addvalmat(1,goofi.exp(goofi.neg(x))));
                                                 }
             public static double[][] nonlinderiv(double[][] x){
             //This functon will find the gradient
                              return goofi.dot(x,goofi.subvalmat(1,x));
                                                      }
             public static void main(String[] args){
             //main
                                double[][] x={{0,0,1},{0,1,1},{1,0,1},{1,1,1}};
                                double[][] y={{0},{1},{1},{0}};                                                         
                                double[][] weight0=goofi.randomize(3,4);//initializing synapses
                                double[][] weight1=goofi.randomize(4,1);//with random values
                                for(int i=0;i<60000;i++){
                                    double[][] layer0=x;//input layer
                                    double[][] layer1=nonlin(goofi.matmul(layer0,weight0));//hidden layer
                                    double[][] layer2=nonlin(goofi.matmul(layer1,weight1));//output layer
                                    double[][] lay2error=goofi.submat(y,layer2);//output layer error
                                    if(i%10000==0){
                                       double k=goofi.mean(goofi.abs(lay2error))*100;//error
                                       System.out.println("output guess");
                                       goofi.printmat(goofi.abs(layer2));//output
                                       System.out.println("Error");
                                       System.out.println(k+"%");
                                                  }
                                    double[][] delta2=goofi.dot(lay2error,nonlinderiv(layer2));//how much change should need in output layer ?
                                    double[][] lay1error=goofi.matmul(delta2,goofi.transpose(weight1));//Error contribution of hidden layer
                                    double[][] delta1=goofi.dot(lay1error,nonlinderiv(layer1));//how much change should need in hidden layer ?
                                    weight1=goofi.addmat(weight1,goofi.matmul(goofi.transpose(layer1),delta2));//updating weights
                                    weight0=goofi.addmat(weight0,goofi.matmul(goofi.transpose(layer0),delta1));//updating weights
                                    

                                                    }
                                       }
                        }
/*


                                  OUTPUT
                                  ------


output guess

[[0.5551902003869719 ]
[0.5934179644199365 ]
[0.6252354872358519 ]
[0.6587842026562442 ]
]
Error
199.53208923339844%
output guess

[[0.014780598866060052 ]
[0.9852139978200263 ]
[0.9783985673299924 ]
[0.02224395859476898 ]
]
Error
7.3411993980407715%
output guess

[[0.005266158302563488 ]
[0.994770883254489 ]
[0.9909469285070046 ]
[0.009656815185114645 ]
]
Error
2.9205162525177%
output guess

[[0.003703262549562145 ]
[0.9963432257761042 ]
[0.9933143314010573 ]
[0.007205166519900109 ]
]
Error
2.125087261199951%
output guess

[[0.002997353993700072 ]
[0.9970634975781982 ]
[0.994450255836842 ]
[0.006012632250263506 ]
]
Error
1.74962317943573%
output guess

[[0.0025796302511078297 ]
[0.997493380668515 ]
[0.9951482926783298 ]
[0.0052733804610744315 ]
]
Error
1.5211337804794312%

*/
