use std::{ops::{Add, Mul, Sub}, collections::{BTreeSet, HashMap}, rc::Rc, cell::RefCell};

enum Operation {
    Mul(Value, Value),
    Add(Value, Value),
    Init,
    Exp(Value),
    Powf(Value, f64)
}


struct ValueInner {
    operation: Operation,
    eval: f64,
    grad: HashMap<String, (f64, u64)>,
}

impl ValueInner {
    fn propagate_gradient(&mut self, label: &str, cur: f64) {
        let Some((grad, vis_cnt)) = self.grad.get_mut(label) else {
            panic!("There is a logical bug somewhere.");
        };
        *grad += cur;
        *vis_cnt -= 1;
        if *vis_cnt > 0 {
            return;
        }
        match &self.operation {
            Operation::Mul(val1, val2) => {
                let eval1 = val1.eval();
                let eval2 = val2.eval();
                val1.value_inner.borrow_mut().propagate_gradient(label, *grad*eval2);
                val2.value_inner.borrow_mut().propagate_gradient(label, *grad*eval1);
            },
            Operation::Add(val1, val2) => {
                val1.value_inner.borrow_mut().propagate_gradient(label, *grad);
                val2.value_inner.borrow_mut().propagate_gradient(label, *grad);
            },
            Operation::Exp(val) => {
                let meval = self.eval;
                val.value_inner.borrow_mut().propagate_gradient(label, *grad*meval);
            }
            
            Operation::Init => (),
            Operation::Powf(val, p) => {
                let d = val.eval().powf(p-1.)*(*p);
                val.value_inner.borrow_mut().propagate_gradient(label, *grad*d);
            },
        }
    }
    fn compute_in_visits(&mut self, label: &str) {
        if let Some((_, vis)) = self.grad.get_mut(label) {
            *vis += 1;
        }
        else {
            self.grad.insert(label.into(), (0., 1));

            let visit = |val: &Value| {
                val.value_inner.borrow_mut().compute_in_visits(label);
            };
            match &self.operation {
                Operation::Mul(l, r) => {
                    visit(l); visit(r);
                },
                Operation::Add(l, r) => {
                    visit(l); visit(r);
                },
                Operation::Init => ()   ,
                Operation::Exp(v) => {
                    visit(v);
                },
                Operation::Powf(v, _) => {
                    visit(v);
                },
            }
        }
    }
}

#[derive(Clone)]
pub struct Value {
    value_inner: Rc<RefCell<ValueInner>>
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

impl Value {
    pub fn new(data: f64) -> Self {
        let value_inner = ValueInner {
            eval: data,
            operation: Operation::Init,
            grad: Default::default()
        };
        Self {
            value_inner: Rc::new(RefCell::new(value_inner))
        }
    }
    pub fn c(&self) -> Self {
        self.clone()
    }
    pub fn compute_gradient(&self, label: &str) {
        self.value_inner.borrow_mut().compute_in_visits(label);
        self.value_inner.borrow_mut().propagate_gradient(label, 1.);
    }
    pub fn eval(&self) -> f64 {
        self.value_inner.borrow().eval
    }
    pub fn grad(&self, label: &str) -> f64 {
        self.value_inner.borrow().grad[label].0
    }
    pub fn tanh(self) -> Value {
        let e2x = (self.c()*2.0.into()).exp();
        (e2x.c()-1.0.into())*(e2x+1.0.into()).powf(-1.)
    }
    pub fn exp(self) -> Value {
        let value_inner = ValueInner {
            eval: self.eval().exp(),
            operation: Operation::Exp(self),
            grad: Default::default()
        };

        Value {
            value_inner: Rc::new(RefCell::new(value_inner))
        } 
    }
    pub fn powf(self, p: f64) -> Value {
        let value_inner = ValueInner {
            eval: self.eval().powf(p),
            operation: Operation::Powf(self, p),
            grad: Default::default()
        };

        Value {
            value_inner: Rc::new(RefCell::new(value_inner))
        }
    }
}

impl Add<Value> for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        let value_inner = ValueInner {
            eval: self.eval()+rhs.eval(),
            operation: Operation::Add(self, rhs),
            grad: Default::default()
        };
        Value {
            value_inner: Rc::new(RefCell::new(value_inner))
        }
    }
}

impl Sub<Value> for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Self::Output {
        self+(rhs*(-1.0).into())
    }
}

impl Mul<Value> for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        let value_inner = ValueInner {
            eval: self.eval()*rhs.eval(),
            operation: Operation::Mul(self, rhs),
            grad: Default::default()
        };
        Value {
            value_inner: Rc::new(RefCell::new(value_inner))
        }
    }
}

