package morpheme

import "testing"

func TestMakeWordID(t *testing.T) {
	id, err := MakeWordID(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if id != 4097 {
		t.Fatalf("want 4097, got %d", id)
	}

	if RootOf(id) != 1 {
		t.Fatalf("want root 1, got %d", RootOf(id))
	}
	if VariantOf(id) != 1 {
		t.Fatalf("want variant 1, got %d", VariantOf(id))
	}
}

func TestCognates(t *testing.T) {
	negative, _ := MakeWordID(1, 1) // EN
	negativo, _ := MakeWordID(1, 2) // PT
	excellent, _ := MakeWordID(2, 1)

	if !Cognates(negative, negativo) {
		t.Fatal("negative and negativo must be cognates")
	}
	if Cognates(negative, excellent) {
		t.Fatal("negative and excellent must NOT be cognates")
	}
}

func TestValidate(t *testing.T) {
	id, _ := MakeWordID(5, 3)
	if err := Validate(id, 5, 3); err != nil {
		t.Fatal(err)
	}
	if err := Validate(id, 5, 4); err == nil {
		t.Fatal("want error for wrong variant")
	}
}

func TestBoundaries(t *testing.T) {
	_, err := MakeWordID(0, 1)
	if err == nil {
		t.Fatal("root_id=0 must be invalid")
	}
	_, err = MakeWordID(1, 0)
	if err == nil {
		t.Fatal("variant=0 must be invalid")
	}
	_, err = MakeWordID(MaxRootID, MaxVariant)
	if err != nil {
		t.Fatalf("max values must be valid: %v", err)
	}
}
