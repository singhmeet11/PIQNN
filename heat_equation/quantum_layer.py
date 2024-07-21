## define the quantum layer
import pennylane as qml
import tensorflow as tf


def quantum_layer(n_qubits, circuit_number):
    # we are making this for one layer only for now
    dev = qml.device("default.qubit", wires=n_qubits)
        
    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        n_layers = weights.shape[0]
        
        # print(weights.shape)
        if circuit_number == "1":
            for j in range(0,n_qubits):
                qml.RX(weights[0,j],wires=j)
                qml.RZ(weights[1,j],wires=j)
                
        if circuit_number == "6":
            for j in range(0,n_qubits):
                qml.RX(weights[0,j],wires=j)
                qml.RZ(weights[1,j],wires=j)
        
            qml.CRZ(weights[2,0],wires = (3,2))
            qml.CRZ(weights[2,1],wires = (3,1))
            qml.CRZ(weights[2,2],wires = (3,0))
            qml.CRZ(weights[2,3],wires = (2,3))
            qml.CRZ(weights[3,0],wires = (2,1))
            qml.CRZ(weights[3,1],wires = (2,0))
            qml.CRZ(weights[3,2],wires = (1,3))
            qml.CRZ(weights[3,3],wires = (1,2))
            qml.CRZ(weights[4,0],wires = (1,0))
            qml.CRZ(weights[4,1],wires = (0,3))
            qml.CRZ(weights[4,2],wires = (0,2))
            qml.CRZ(weights[4,3],wires = (0,1))

            for j in range(0,n_qubits):
                qml.RX(weights[5,j],wires=j)
                qml.RZ(weights[6,j],wires=j)
            
        if circuit_number == "7":
            for j in range(0,n_qubits):
                qml.RX(weights[0,j],wires=j)
                qml.RZ(weights[1,j],wires=j)
            
            qml.CRZ(weights[2,0], wires=(1,0))
            qml.CRZ(weights[2,1], wires=(3,2))
            
            for j in range(0,n_qubits):
                qml.RX(weights[3,j],wires=j)
                qml.RZ(weights[4,j],wires=j)
                
            qml.CRZ(weights[2,3], wires=(2,1))

        if circuit_number == "14":
            for j in range(0,n_qubits):
                qml.RY(weights[0,j],wires=j)
            qml.CRX(weights[1,0],wires = (3,0))
            qml.CRX(weights[1,1],wires = (2,3))
            qml.CRX(weights[1,2],wires = (1,2))
            qml.CRX(weights[1,3],wires = (0,1))
            for j in range(0,n_qubits):
                qml.RY(weights[2,j],wires=j)
            qml.CRX(weights[3,0],wires = (3,2))
            qml.CRX(weights[3,1],wires = (0,3))
            qml.CRX(weights[3,2],wires = (1,0))
            qml.CRX(weights[3,3],wires = (2,1))
            
        measures = [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        return measures
    
    if circuit_number =="1":
        depth = 2
    if circuit_number =="6":
        depth = 7
    if circuit_number == "7":
        depth = 5
    if circuit_number == "14":
        depth = 4
    
    weight_shapes = {"weights": (depth, n_qubits)}

    qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)

    return qlayer
